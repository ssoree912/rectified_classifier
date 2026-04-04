import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    if t.ndim != 1:
        t = t.view(t.shape[0])
    shape = [t.shape[0]] + [1] * (ndim - 1)
    return t.view(*shape)


def interpolate_latents(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = _broadcast_t(t.to(device=z0.device, dtype=z0.dtype), z0.ndim)
    return (1.0 - t_view) * z0 + t_view * z1


def velocity_target(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    return z1 - z0


class LatentVelocityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        depth: int = 3,
        t_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.t_dim = int(t_dim)
        self.dropout = float(dropout)
        self.model_kind = "velocity_mlp"

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.t_dim),
            nn.GELU(),
            nn.Linear(self.t_dim, self.t_dim),
            nn.GELU(),
        )

        layers = []
        in_dim = self.input_dim + self.t_dim
        for _ in range(self.depth - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 2:
            raise ValueError(f"LatentVelocityMLP expects vector inputs (B, D), got {tuple(z_t.shape)}")
        if t.ndim != 1:
            t = t.view(t.shape[0])
        t_emb = self.time_mlp(t[:, None].to(device=z_t.device, dtype=z_t.dtype))
        x = torch.cat([self.input_norm(z_t), t_emb], dim=1)
        return self.net(x)


class ResidualTimeConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x + t_emb
        h = self.conv1(F.gelu(self.norm1(h)))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class TokenMapVelocityCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        t_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.t_dim = int(t_dim)
        self.dropout = float(dropout)
        self.model_kind = "token_map_velocity_cnn"

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.t_dim),
            nn.GELU(),
            nn.Linear(self.t_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualTimeConvBlock(self.hidden_dim, dropout=self.dropout) for _ in range(self.depth)]
        )
        self.output_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.output_proj = nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 4:
            raise ValueError(
                f"TokenMapVelocityCNN expects map inputs (B, C, H, W), got {tuple(z_t.shape)}"
            )
        if t.ndim != 1:
            t = t.view(t.shape[0])
        h = self.input_proj(z_t)
        t_emb = self.time_mlp(t[:, None].to(device=z_t.device, dtype=z_t.dtype))[:, :, None, None]
        for block in self.blocks:
            h = block(h, t_emb)
        h = F.gelu(self.output_norm(h))
        return self.output_proj(h)


def euler_transport(model: nn.Module, z0: torch.Tensor, num_steps: int = 4) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    z = z0.clone()
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full((z.shape[0],), float(step) / float(num_steps), device=z.device, dtype=torch.float32)
        v = model(z, t)
        z = z + dt * v
    return z


def build_latent_velocity_model(
    input_dim: int,
    is_spatial: bool,
    hidden_dim: int,
    depth: int,
    t_dim: int,
    dropout: float = 0.0,
    model_kind: str = "auto",
):
    kind = model_kind
    if kind == "auto":
        kind = "token_map_velocity_cnn" if is_spatial else "velocity_mlp"

    if kind == "token_map_velocity_cnn":
        model = TokenMapVelocityCNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            t_dim=t_dim,
            dropout=dropout,
        )
        return model, kind

    if kind == "velocity_mlp":
        model = LatentVelocityMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            t_dim=t_dim,
            dropout=dropout,
        )
        return model, kind

    raise ValueError(f"Unsupported model_kind: {model_kind}")


def build_latent_velocity_from_checkpoint(checkpoint, input_dim=None, is_spatial=None):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        ckpt_args = checkpoint.get("args", {}) or {}
        inferred_input_dim = checkpoint.get("feature_dim") or ckpt_args.get("feature_dim")
        inferred_hidden_dim = checkpoint.get("hidden_dim") or ckpt_args.get("hidden_dim")
        inferred_depth = checkpoint.get("depth") or ckpt_args.get("depth")
        inferred_t_dim = checkpoint.get("t_dim") or ckpt_args.get("t_dim")
        inferred_dropout = checkpoint.get("dropout") or ckpt_args.get("dropout") or 0.0
        inferred_is_spatial = checkpoint.get("is_spatial")
        if inferred_is_spatial is None:
            inferred_is_spatial = ckpt_args.get("is_spatial")
        inferred_model_kind = checkpoint.get("model_kind") or ckpt_args.get("model_kind") or "auto"
        integration_steps = checkpoint.get("integration_steps") or ckpt_args.get("integration_steps") or 4
    else:
        state_dict = checkpoint
        inferred_input_dim = None
        inferred_hidden_dim = None
        inferred_depth = None
        inferred_t_dim = None
        inferred_dropout = 0.0
        inferred_is_spatial = None
        inferred_model_kind = "auto"
        integration_steps = 4

    input_dim = int(input_dim or inferred_input_dim or 0)
    if input_dim <= 0:
        raise ValueError("input_dim must be provided either explicitly or inside the checkpoint metadata.")

    if is_spatial is None:
        is_spatial = bool(inferred_is_spatial)

    model, resolved_kind = build_latent_velocity_model(
        input_dim=input_dim,
        is_spatial=is_spatial,
        hidden_dim=int(inferred_hidden_dim or 128),
        depth=int(inferred_depth or 4),
        t_dim=int(inferred_t_dim or 64),
        dropout=float(inferred_dropout or 0.0),
        model_kind=inferred_model_kind,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, {
        "model_kind": resolved_kind,
        "input_dim": input_dim,
        "hidden_dim": int(inferred_hidden_dim or 128),
        "depth": int(inferred_depth or 4),
        "t_dim": int(inferred_t_dim or 64),
        "dropout": float(inferred_dropout or 0.0),
        "is_spatial": bool(is_spatial),
        "integration_steps": int(integration_steps),
    }
