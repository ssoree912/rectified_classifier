import torch.nn as nn


class SpatialTransformer(nn.Module):
    """
    Minimal fallback for openaimodel_medical imports.
    In this project path, UNet is instantiated with use_spatial_transformer=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x, context=None):
        return self.identity(x)
