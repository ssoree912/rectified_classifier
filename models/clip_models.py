from .clip import clip
from PIL import Image
import torch.nn as nn
import torch
import os
from models.transformer_attention import TransformerAttention
import torchvision.transforms.functional as TF
from .clip.model import VisionTransformer
from .mlp import MLP
from .latent_rectifier import clip_input_size

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "ViT-L/14-penultimate": 1024,
}


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


class CLIPModel(nn.Module):
    """UFD"""

    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)


class CLIPModelPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModelPenultimateLayer, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        self.register_hook()
        self.fc = nn.Linear(CHANNELS[name + "-penultimate"], num_classes)

    def register_hook(self):
        def hook(module, inputs, output):
            self.features = torch.clone(output)

        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return

    def forward(self, x):
        self.model.encode_image(x)
        return self.fc(self.features)


class CLIPModelRectifyDiscrepancyAttention(nn.Module):
    """
    Two-view attention classifier:
      1) original x
      2) discrepancy delta = |SR(D(x)) - R(SR(D(x)))|
    """

    def __init__(
        self,
        name,
        num_classes=1,
        rectifier=None,
        input_is_clip_normalized=True,
        freeze_rectifier=True,
    ):
        super(CLIPModelRectifyDiscrepancyAttention, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.input_is_clip_normalized = input_is_clip_normalized
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.register_hook()
        self.rectifier = rectifier
        self.sr_cache_root = None
        self.sr_cache_input_root = None
        self.current_paths = None

        feature_dim = self._feature_dim()
        self.attention_head = TransformerAttention(feature_dim, 2, last_dim=num_classes)

        for _, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()

        clip_mean = torch.tensor(MEAN["clip"]).view(1, 3, 1, 1)
        clip_std = torch.tensor(STD["clip"]).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", clip_mean, persistent=False)
        self.register_buffer("clip_std", clip_std, persistent=False)

        if self.rectifier is not None and freeze_rectifier:
            for p in self.rectifier.parameters():
                p.requires_grad = False
            self.rectifier.eval()

    def _feature_dim(self):
        mapped = CHANNELS.get(self.name + "-penultimate") or CHANNELS.get(self.name)
        if mapped is not None:
            return mapped
        size = clip_input_size(self.name)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, size, size)
            _ = self.model.encode_image(dummy)
            feat = self.features
            if feat.ndim == 3:
                feat = feat[:, 0, :]
        return int(feat.shape[-1])

    def register_hook(self):
        def hook(module, inputs, output):
            self.features = torch.clone(output)

        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return

    def set_rectify_modules(self, rectifier, freeze_rectifier=True):
        self.rectifier = rectifier.to(self.clip_mean.device)
        if freeze_rectifier:
            for p in self.rectifier.parameters():
                p.requires_grad = False
            self.rectifier.eval()

    def set_sr_cache(self, sr_cache_root=None, sr_cache_input_root=None):
        self.sr_cache_root = sr_cache_root
        self.sr_cache_input_root = sr_cache_input_root

    def set_current_paths(self, paths):
        self.current_paths = list(paths) if paths is not None else None

    def _to_image_space(self, x):
        if self.input_is_clip_normalized:
            x = x * self.clip_std + self.clip_mean
        return x.clamp(0.0, 1.0)

    def _to_clip_space(self, x):
        return (x - self.clip_mean) / self.clip_std

    def _resolve_cached_path(self, src_path):
        if self.sr_cache_root is None or self.sr_cache_input_root is None:
            return None
        rel = os.path.relpath(src_path, self.sr_cache_input_root)
        base = os.path.join(self.sr_cache_root, rel)
        if os.path.exists(base):
            return base
        stem, _ = os.path.splitext(base)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            cand = stem + ext
            if os.path.exists(cand):
                return cand
        return None

    def _load_cached_sr_batch(self, paths, target_hw, device, dtype):
        tensors = []
        for path in paths:
            cache_path = self._resolve_cached_path(path)
            if cache_path is None:
                return None
            with Image.open(cache_path) as img:
                tensors.append(TF.to_tensor(img.convert("RGB")))
        x_sr = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
        if x_sr.shape[-2:] != target_hw:
            x_sr = torch.nn.functional.interpolate(
                x_sr, size=target_hw, mode="bilinear", align_corners=False
            )
        return x_sr.clamp(0.0, 1.0)

    @torch.no_grad()
    def _make_delta(self, x):
        if self.rectifier is None:
            raise RuntimeError("rectifier must be set before forward().")
        if self.sr_cache_root is None or self.sr_cache_input_root is None:
            raise RuntimeError(
                "SR cache is required. Set --sr_cache_root and --sr_cache_input_root."
            )

        x_img = self._to_image_space(x)
        if next(self.rectifier.parameters()).device != x.device:
            self.rectifier = self.rectifier.to(x.device)

        if self.current_paths is None or len(self.current_paths) != x.shape[0]:
            raise RuntimeError("Current image paths are required to resolve SR cache.")
        x_sr = self._load_cached_sr_batch(
            self.current_paths,
            target_hw=x_img.shape[-2:],
            device=x.device,
            dtype=x.dtype,
        )
        if x_sr is None:
            raise RuntimeError("Missing SR cache file for one or more images in batch.")

        x_hat = self.rectifier(x_sr)
        delta = torch.abs(x_sr - x_hat).clamp(0.0, 1.0)
        return self._to_clip_space(delta).to(dtype=x.dtype)

    @torch.no_grad()
    def _encode_penultimate(self, x):
        _ = self.model.encode_image(x)
        feat = self.features
        if feat.ndim == 3:
            feat = feat[:, 0, :]
        return feat

    def forward(self, x, return_feature=False):
        with torch.no_grad():
            delta = self._make_delta(x)
            f_orig = self._encode_penultimate(x).float()
            f_delta = self._encode_penultimate(delta).float()
        self.current_paths = None
        view_features = torch.stack([f_orig, f_delta], dim=1)
        fused = self.attention_head(view_features)
        if return_feature:
            return fused, view_features
        return fused


class CLIPModelLatentRectifyAttention(CLIPModelRectifyDiscrepancyAttention):
    """
    Two-view attention classifier in latent space:
      1) original CLIP penultimate feature
      2) latent discrepancy or rectified latent derived from SR(D(x))
    """

    def __init__(
        self,
        name,
        num_classes=1,
        latent_rectifier=None,
        input_is_clip_normalized=True,
        freeze_rectifier=True,
        latent_view_mode="delta",
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            rectifier=None,
            input_is_clip_normalized=input_is_clip_normalized,
            freeze_rectifier=False,
        )
        self.latent_rectifier = None
        self.latent_view_mode = latent_view_mode
        if latent_rectifier is not None:
            self.set_latent_rectify_modules(latent_rectifier, freeze_rectifier=freeze_rectifier)

    def set_latent_rectify_modules(self, rectifier, freeze_rectifier=True):
        self.latent_rectifier = rectifier.to(self.clip_mean.device)
        if freeze_rectifier:
            for p in self.latent_rectifier.parameters():
                p.requires_grad = False
            self.latent_rectifier.eval()

    @torch.no_grad()
    def _make_latent_views(self, x):
        if self.latent_rectifier is None:
            raise RuntimeError("latent rectifier must be set before forward().")
        if self.sr_cache_root is None or self.sr_cache_input_root is None:
            raise RuntimeError(
                "SR cache is required. Set --sr_cache_root and --sr_cache_input_root."
            )
        if self.current_paths is None or len(self.current_paths) != x.shape[0]:
            raise RuntimeError("Current image paths are required to resolve SR cache.")

        if next(self.latent_rectifier.parameters()).device != x.device:
            self.latent_rectifier = self.latent_rectifier.to(x.device)

        x_img = self._to_image_space(x)
        x_sr = self._load_cached_sr_batch(
            self.current_paths,
            target_hw=x_img.shape[-2:],
            device=x.device,
            dtype=x_img.dtype,
        )
        if x_sr is None:
            raise RuntimeError("Missing SR cache file for one or more images in batch.")

        x_sr_clip = self._to_clip_space(x_sr).to(dtype=x.dtype)
        f_orig = self._encode_penultimate(x).float()
        f_sr = self._encode_penultimate(x_sr_clip).float()
        f_hat = self.latent_rectifier(f_sr.float()).float()

        if self.latent_view_mode == "rectified":
            second = f_hat
        elif self.latent_view_mode == "sr":
            second = f_sr
        else:
            second = torch.abs(f_sr - f_hat)
        return f_orig, second

    def forward(self, x, return_feature=False):
        with torch.no_grad():
            f_orig, f_aux = self._make_latent_views(x)
        self.current_paths = None
        view_features = torch.stack([f_orig, f_aux], dim=1)
        fused = self.attention_head(view_features)
        if return_feature:
            return fused, view_features
        return fused
