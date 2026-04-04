import math

import torch
import torch.nn as nn

from .clip import clip


def clip_input_size(name: str) -> int:
    return 336 if "@336px" in name else 224


def _square_grid_size(num_tokens: int) -> int:
    grid_size = int(math.isqrt(int(num_tokens)))
    if grid_size * grid_size != int(num_tokens):
        raise ValueError(f"Patch token count {num_tokens} is not a perfect square.")
    return grid_size


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def extract_clip_visual_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
    if any(not hasattr(visual, name) for name in required):
        raise ValueError("Token extraction is only supported for CLIP ViT backbones.")

    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)

    positional = visual.positional_embedding.to(x.dtype)
    if positional.shape[0] != x.shape[1]:
        raise ValueError(
            f"Input resolution is incompatible with the CLIP positional embedding: tokens={x.shape[1]}, positional={positional.shape[0]}"
        )
    x = x + positional
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    _, x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    return x


def tokens_to_cls_vector(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError(f"Expected token tensor with shape (B, N, D), got {tuple(tokens.shape)}")
    return tokens[:, 0, :].contiguous()


def tokens_to_gap_vector(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError(f"Expected CLS + patch tokens, got shape={tuple(tokens.shape)}")
    return tokens[:, 1:, :].mean(dim=1).contiguous()


def tokens_to_patch_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with shape (B, N, D), got {tuple(tokens.shape)}")
    if tokens.shape[1] < 2:
        raise ValueError(f"Expected CLS + patch tokens, got shape={tuple(tokens.shape)}")

    patch_tokens = tokens[:, 1:, :]
    batch_size, num_tokens, channels = patch_tokens.shape
    grid_size = _square_grid_size(num_tokens)
    patch_map = patch_tokens.reshape(batch_size, grid_size, grid_size, channels)
    return patch_map.permute(0, 3, 1, 2).contiguous()


class CLIPPenultimateEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        tokens = extract_clip_visual_tokens(self.model, x)
        return tokens_to_cls_vector(tokens)

    @torch.no_grad()
    def encode_gap_pooled(self, x: torch.Tensor) -> torch.Tensor:
        tokens = extract_clip_visual_tokens(self.model, x)
        return tokens_to_gap_vector(tokens)

    @torch.no_grad()
    def encode_token_map(self, x: torch.Tensor) -> torch.Tensor:
        tokens = extract_clip_visual_tokens(self.model, x)
        return tokens_to_patch_map(tokens)

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor, latent_kind: str = "cls") -> torch.Tensor:
        if latent_kind == "token_map":
            return self.encode_token_map(x)
        if latent_kind == "gap":
            return self.encode_gap_pooled(x)
        if latent_kind == "cls":
            return self.encode_penultimate(x)
        raise ValueError(f"Unsupported latent_kind: {latent_kind}")

    @torch.no_grad()
    def infer_feature_dim(self) -> int:
        size = clip_input_size(self.name)
        device = next(self.model.parameters()).device
        dummy = torch.zeros(1, 3, size, size, device=device)
        z = self.encode_penultimate(dummy)
        return int(z.shape[-1])

    @torch.no_grad()
    def infer_token_map_shape(self):
        device = next(self.model.parameters()).device
        size = clip_input_size(self.name)
        dummy = torch.zeros(1, 3, size, size, device=device)
        z = self.encode_token_map(dummy)
        return int(z.shape[1]), int(z.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_penultimate(x)


class LatentRectifierMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        depth: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim or max(input_dim * 2, 512))
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.input_norm = nn.LayerNorm(self.input_dim)
        layers = []
        in_dim = self.input_dim
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(self.input_norm(z))
        if self.residual:
            return z + delta
        return delta


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout2d(dropout),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TokenMapRectifierCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualConvBlock(self.hidden_dim, dropout=self.dropout) for _ in range(self.depth)])
        self.output_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.output_proj = nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        h = torch.nn.functional.gelu(h)
        delta = self.output_proj(h)
        if self.residual:
            return z + delta
        return delta


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def build_latent_rectifier_from_checkpoint(checkpoint, input_dim=None, hidden_dim=None, depth=None):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        ckpt_args = checkpoint.get("args", {}) or {}
        inferred_input_dim = checkpoint.get("feature_dim") or ckpt_args.get("feature_dim")
        inferred_hidden_dim = checkpoint.get("hidden_dim") or ckpt_args.get("hidden_dim")
        inferred_depth = checkpoint.get("depth") or ckpt_args.get("depth")
        rectifier_kind = checkpoint.get("rectifier_kind") or ckpt_args.get("rectifier_kind") or "mlp"
    else:
        state_dict = checkpoint
        ckpt_args = {}
        inferred_input_dim = None
        inferred_hidden_dim = None
        inferred_depth = None
        rectifier_kind = "mlp"

    state_dict = _strip_module_prefix(state_dict)

    input_dim = int(input_dim or inferred_input_dim or 0)
    if input_dim <= 0:
        raise ValueError("input_dim must be provided either explicitly or inside the checkpoint metadata.")

    hidden_dim = int(hidden_dim or inferred_hidden_dim or 128)
    depth = int(depth or inferred_depth or 4)

    if rectifier_kind == "token_map_cnn":
        model = TokenMapRectifierCNN(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth)
    else:
        rectifier_kind = "mlp"
        model = LatentRectifierMLP(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth)
    model.load_state_dict(state_dict, strict=True)
    return model, {
        "rectifier_kind": rectifier_kind,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "checkpoint_args": ckpt_args,
    }
