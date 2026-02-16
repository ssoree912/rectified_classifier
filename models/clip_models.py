from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
import os
from models.transformer_attention import TransformerAttention
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .clip.model import VisionTransformer
from .mlp import MLP

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14-penultimate" : 1024
}


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

class CLIPModel(nn.Module):
    """UFD"""
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )


    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)


class CLIPModelPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModelPenultimateLayer, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.register_hook()
        self.fc = nn.Linear(CHANNELS[name+"-penultimate"], num_classes)

    def register_hook(self):
        
        def hook(module, input, output):
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
        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        self.register_hook()
        self.rectifier = rectifier
        self.sr_cache_root = None
        self.sr_cache_input_root = None
        self.current_paths = None

        feature_dim = CHANNELS[name + "-penultimate"] if (name + "-penultimate") in CHANNELS else CHANNELS[name]
        self.attention_head = TransformerAttention(feature_dim, 2, last_dim=num_classes)

        for name, param in self.model.named_parameters():
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

    def register_hook(self):
        
        def hook(module, input, output):
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
        for p in paths:
            cache_path = self._resolve_cached_path(p)
            if cache_path is None:
                return None
            img = Image.open(cache_path).convert("RGB")
            tensors.append(TF.to_tensor(img))
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
            f_orig = self._encode_penultimate(x)
            f_delta = self._encode_penultimate(delta)
        self.current_paths = None
        view_features = torch.stack([f_orig, f_delta], dim=1)
        fused = self.attention_head(view_features)
        if return_feature:
            return fused, view_features
        return fused
