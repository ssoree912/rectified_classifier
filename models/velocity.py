import torch
import torch.nn as nn
from models.openaimodel_medical import UNetModel as MedicalUNet


def _default_t(x: torch.Tensor) -> torch.Tensor:
    """Single-step rectification uses a fixed time token t=0."""
    return torch.zeros(x.size(0), 1, device=x.device, dtype=torch.float32)


class StageVelocityUNet(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int = 128,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.unet = MedicalUNet(
            image_size=None,
            in_channels=c_in,
            model_channels=c_hidden,
            out_channels=c_in,
            num_res_blocks=num_res_blocks,
            attention_resolutions=[],
            channel_mult=(1, 1, 2),
            conv_resample=True, dims=2, dropout=0.0,
            num_classes=None, use_checkpoint=False, use_fp16=False,
            num_heads=num_heads, num_head_channels=-1, num_heads_upsample=-1,
            use_scale_shift_norm=True, resblock_updown=False
        )

    def forward(self, x, t=None):
        """
        Rectifier mode (recommended):
            x_hat = x + UNet(x, t=0)
        """
        if t is None:
            t = _default_t(x)
        elif t.ndim == 1:
            t = t.unsqueeze(1)
        t = t.to(device=x.device, dtype=torch.float32)

        delta = self.unet(x, t=t)
        if self.residual:
            return x + delta
        return delta


class RectifierUNet(StageVelocityUNet):
    def __init__(self, c_in: int = 3, c_hidden: int = 128, num_res_blocks: int = 2, num_heads: int = 4):
        super().__init__(
            c_in=c_in,
            c_hidden=c_hidden,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,
            residual=True,
        )


def discrepancy_from_sr(x_sr: torch.Tensor, x_hat: torch.Tensor):
    """
    Returns:
        delta_map: |x_sr - x_hat|
        score: per-sample scalar discrepancy
        """
    delta_map = torch.abs(x_sr - x_hat)
    score = delta_map.mean(dim=[1, 2, 3])
    return delta_map, score

class Velocity3Stage(nn.Module):
    def __init__(self, c_list, cfg):
        super().__init__()
        self.stage1 = StageVelocityUNet(c_list[0], cfg['s1_hidden'], cfg['s1_resblocks'], 4, cfg.get('residual', True))
        self.stage2 = StageVelocityUNet(c_list[1], cfg['s2_hidden'], cfg['s2_resblocks'], 4, cfg.get('residual', True))
        self.stage3 = StageVelocityUNet(c_list[2], cfg['s3_hidden'], cfg['s3_resblocks'], 4, cfg.get('residual', True))

    def forward(self, xs, t=None):
        x1, x2, x3 = xs
        d1 = self.stage1(x1, t)
        d2 = self.stage2(x2, t)
        d3 = self.stage3(x3, t)
        return d1, d2, d3
