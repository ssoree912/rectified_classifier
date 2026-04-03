import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt)
        print(f"using {self.model.__class__.__name__}")

        if opt.head_type == "fc":
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        elif opt.head_type == "attention":
            for _, params in self.model.attention_head.named_parameters():
                torch.nn.init.normal_(params, 0.0, opt.init_gain)

        if opt.resume_path is not None:
            state_dict = torch.load(opt.resume_path)
            if self.opt.fix_backbone:
                if self.opt.head_type == "attention" or opt.head_type == "crossattention":
                    self.model.attention_head.load_state_dict(state_dict)
                else:
                    self.model.fc.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

        if opt.fix_backbone:
            params = []
            if opt.head_type == "fc":
                for name, param in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias":
                        params.append(param)
                    else:
                        param.requires_grad = False
            elif opt.head_type == "mlp":
                for param in self.model.mlp.parameters():
                    params.append(param)
            elif opt.head_type == "attention" or opt.head_type == "crossattention":
                for param in self.model.attention_head.parameters():
                    params.append(param)
            elif opt.head_type == "transformer":
                params = [
                    {'params': self.model.transformer_block.parameters()},
                    {'params': self.model.fc.parameters()},
                ]
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model.to(opt.gpu_ids[0])
        self._attach_rectify_modules_if_needed()

    def _attach_rectify_modules_if_needed(self):
        has_pixel = hasattr(self.model, "set_rectify_modules")
        has_latent = hasattr(self.model, "set_latent_rectify_modules")
        if not has_pixel and not has_latent:
            return

        if self.opt.rectifier_ckpt is None:
            raise ValueError(
                "rectifier_ckpt is required for discrepancy attention model. "
                "Set --rectifier_ckpt /path/to/rectifier.pth"
            )
        if self.opt.sr_cache_root is None or self.opt.sr_cache_input_root is None:
            raise ValueError(
                "SR cache is required. Set both --sr_cache_root and --sr_cache_input_root."
            )

        if getattr(self.opt, "rectifier_mode", "pixel") == "latent":
            self._attach_latent_rectifier()
        else:
            self._attach_pixel_rectifier()

        if hasattr(self.model, "set_sr_cache"):
            self.model.set_sr_cache(
                sr_cache_root=self.opt.sr_cache_root,
                sr_cache_input_root=self.opt.sr_cache_input_root,
            )

    def _attach_pixel_rectifier(self):
        if not hasattr(self.model, "set_rectify_modules"):
            raise ValueError("Current model does not support pixel-space rectifier attachment.")

        from models.velocity import RectifierUNet

        device = f"cuda:{self.opt.gpu_ids[0]}" if len(self.opt.gpu_ids) > 0 else "cpu"
        self.rectifier = RectifierUNet(c_in=3)
        state_dict = torch.load(self.opt.rectifier_ckpt, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.rectifier.load_state_dict(state_dict, strict=True)
        self.rectifier.to(device).eval()
        self.model.set_rectify_modules(self.rectifier, freeze_rectifier=True)
        print(f"Attached pixel-space SR + rectifier from: {self.opt.rectifier_ckpt}")

    def _attach_latent_rectifier(self):
        if not hasattr(self.model, "set_latent_rectify_modules"):
            raise ValueError("Current model does not support latent-space rectifier attachment.")

        from models.latent_rectifier import build_latent_rectifier_from_checkpoint

        device = f"cuda:{self.opt.gpu_ids[0]}" if len(self.opt.gpu_ids) > 0 else "cpu"
        checkpoint = torch.load(self.opt.rectifier_ckpt, map_location="cpu")
        input_dim = None
        if hasattr(self.model, "attention_head") and hasattr(self.model.attention_head, "query"):
            input_dim = self.model.attention_head.query.in_features
        self.rectifier, meta = build_latent_rectifier_from_checkpoint(
            checkpoint,
            input_dim=input_dim,
            hidden_dim=None,
            depth=None,
        )
        self.rectifier.to(device).eval()
        self.model.set_latent_rectify_modules(self.rectifier, freeze_rectifier=True)
        if hasattr(self.model, "latent_view_mode"):
            self.model.latent_view_mode = getattr(self.opt, "latent_view_mode", "delta")
        print(
            f"Attached latent-space rectifier from: {self.opt.rectifier_ckpt} "
            f"(input_dim={meta['input_dim']}, hidden_dim={meta['hidden_dim']}, depth={meta['depth']}, view={getattr(self.opt, 'latent_view_mode', 'delta')})"
        )

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        self.image_paths = input[2] if len(input) > 2 else None

    def forward(self):
        if hasattr(self.model, "set_current_paths"):
            self.model.set_current_paths(self.image_paths)
        self.output = self.model(self.input)
        self.output = self.output

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
