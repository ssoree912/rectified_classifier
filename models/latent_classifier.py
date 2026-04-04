import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_attention import TransformerAttention


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _attention_head_count(hidden_dim: int) -> int:
    for heads in (8, 4, 2, 1):
        if hidden_dim % heads == 0:
            return heads
    return 1


def _build_2d_sincos_pos_embed(height: int, width: int, dim: int, device, dtype):
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin/cos embedding, got {dim}")

    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)

    quarter_dim = dim // 4
    omega = torch.arange(quarter_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(quarter_dim - 1, 1)))

    y_proj = coords[:, 0:1] * omega.unsqueeze(0)
    x_proj = coords[:, 1:2] * omega.unsqueeze(0)
    pos = torch.cat([torch.sin(y_proj), torch.cos(y_proj), torch.sin(x_proj), torch.cos(x_proj)], dim=1)
    return pos.to(dtype=dtype)


class LatentPairAttentionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_tokens: int = 2, num_classes: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.token_norm = nn.LayerNorm(self.input_dim)
        self.attention_head = TransformerAttention(self.input_dim, self.num_tokens, last_dim=num_classes)
        self.classifier_kind = 'vector_attention'

    @staticmethod
    def _as_feature(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            return F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        if z.ndim == 2:
            return z
        raise ValueError(f"Unsupported latent batch shape: {tuple(z.shape)}")

    def forward(self, z_orig: torch.Tensor, z_aux: torch.Tensor, return_feature: bool = False):
        f_orig = self.token_norm(self._as_feature(z_orig))
        f_aux = self.token_norm(self._as_feature(z_aux))
        x = torch.stack([f_orig, f_aux], dim=1)
        logits = self.attention_head(x)
        if return_feature:
            return logits, x
        return logits


class MapSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualMapBlock(nn.Module):
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


class LatentPairMapClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        num_classes: int = 1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.classifier_kind = 'map_cnn'

        self.orig_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.aux_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.fuse = nn.Conv2d(self.hidden_dim * 3, self.hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualMapBlock(self.hidden_dim, dropout=self.dropout) for _ in range(self.depth)])
        self.out_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )

    def forward(self, z_orig: torch.Tensor, z_aux: torch.Tensor, return_feature: bool = False):
        if z_orig.ndim != 4 or z_aux.ndim != 4:
            raise ValueError(
                f"LatentPairMapClassifier expects map tensors of shape (B, C, H, W), got {tuple(z_orig.shape)} and {tuple(z_aux.shape)}"
            )
        h_orig = self.orig_proj(z_orig)
        h_aux = self.aux_proj(z_aux)
        h = torch.cat([h_orig, h_aux, h_orig - h_aux], dim=1)
        h = self.fuse(h)
        for block in self.blocks:
            h = block(h)
        h = F.gelu(self.out_norm(h))
        pooled = self.pool(h)
        logits = self.head(pooled)
        if return_feature:
            return logits, pooled
        return logits


class LatentPairMapAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        num_classes: int = 1,
        num_heads: int = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.num_heads = int(num_heads or _attention_head_count(self.hidden_dim))
        self.classifier_kind = 'map_attention'

        self.orig_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.aux_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.view_embeddings = nn.Parameter(torch.zeros(2, 1, self.hidden_dim))
        self.blocks = nn.ModuleList(
            [MapSelfAttentionBlock(self.hidden_dim, self.num_heads, dropout=self.dropout) for _ in range(self.depth)]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.view_embeddings, std=0.02)

    def forward(self, z_orig: torch.Tensor, z_aux: torch.Tensor, return_feature: bool = False):
        if z_orig.ndim != 4 or z_aux.ndim != 4:
            raise ValueError(
                f"LatentPairMapAttentionClassifier expects map tensors of shape (B, C, H, W), got {tuple(z_orig.shape)} and {tuple(z_aux.shape)}"
            )
        if z_orig.shape != z_aux.shape:
            raise ValueError(f"Shape mismatch: {tuple(z_orig.shape)} vs {tuple(z_aux.shape)}")

        batch_size, _, height, width = z_orig.shape
        orig_tokens = z_orig.flatten(2).transpose(1, 2)
        aux_tokens = z_aux.flatten(2).transpose(1, 2)
        pos_embed = _build_2d_sincos_pos_embed(height, width, self.hidden_dim, z_orig.device, orig_tokens.dtype)
        pos_embed = pos_embed.unsqueeze(0)

        orig_tokens = self.orig_proj(orig_tokens) + pos_embed + self.view_embeddings[0]
        aux_tokens = self.aux_proj(aux_tokens) + pos_embed + self.view_embeddings[1]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, orig_tokens, aux_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_feature = x[:, 0, :]
        logits = self.head(cls_feature)
        if return_feature:
            return logits, x
        return logits


def build_latent_pair_classifier(
    input_dim: int,
    is_spatial: bool = False,
    classifier_kind: str = 'auto',
    map_hidden_dim: int = 128,
    map_depth: int = 4,
    map_dropout: float = 0.0,
):
    kind = classifier_kind
    if kind == 'auto':
        kind = 'map_attention' if is_spatial else 'vector_attention'

    if kind == 'map_cnn':
        model = LatentPairMapClassifier(
            input_dim=input_dim,
            hidden_dim=map_hidden_dim,
            depth=map_depth,
            dropout=map_dropout,
        )
        return model, kind

    if kind == 'map_attention':
        model = LatentPairMapAttentionClassifier(
            input_dim=input_dim,
            hidden_dim=map_hidden_dim,
            depth=map_depth,
            dropout=map_dropout,
        )
        return model, kind

    if kind == 'vector_attention':
        model = LatentPairAttentionClassifier(input_dim=input_dim)
        return model, kind

    raise ValueError(f"Unsupported classifier_kind: {classifier_kind}")
