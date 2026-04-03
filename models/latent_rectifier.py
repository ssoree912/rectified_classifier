import torch
import torch.nn as nn

from .clip import clip


def clip_input_size(name: str) -> int:
    return 336 if "@336px" in name else 224


class CLIPPenultimateEncoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.features = None
        self._register_hook()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _register_hook(self):
        def hook(module, inputs, output):
            self.features = torch.clone(output)

        for module_name, module in self.model.visual.named_children():
            if module_name == "ln_post":
                module.register_forward_hook(hook)
                return
        raise ValueError(f"Could not register penultimate hook for CLIP visual backbone: {self.name}")

    @torch.no_grad()
    def encode_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model.encode_image(x)
        if self.features is None:
            raise RuntimeError("CLIP penultimate features were not populated by the forward hook.")
        feat = self.features
        if feat.ndim == 3:
            feat = feat[:, 0, :]
        return feat

    @torch.no_grad()
    def infer_feature_dim(self) -> int:
        device = next(self.model.parameters()).device
        size = clip_input_size(self.name)
        dummy = torch.zeros(1, 3, size, size, device=device)
        return int(self.encode_penultimate(dummy).shape[-1])

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
        inferred_hidden_dim = ckpt_args.get("hidden_dim")
        inferred_depth = ckpt_args.get("depth")
    else:
        state_dict = checkpoint
        ckpt_args = {}
        inferred_input_dim = None
        inferred_hidden_dim = None
        inferred_depth = None

    state_dict = _strip_module_prefix(state_dict)

    input_dim = int(input_dim or inferred_input_dim or 0)
    if input_dim <= 0:
        raise ValueError("input_dim must be provided either explicitly or inside the checkpoint metadata.")

    hidden_dim = int(hidden_dim or inferred_hidden_dim or max(input_dim * 2, 512))
    depth = int(depth or inferred_depth or 3)
    model = LatentRectifierMLP(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth)
    model.load_state_dict(state_dict, strict=True)
    return model, {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "checkpoint_args": ckpt_args,
    }
