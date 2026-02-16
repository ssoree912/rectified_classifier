import torch.nn as nn


class ResidualRectifierCNN(nn.Module):
    """Lightweight residual denoiser: r_noisy -> r_clean."""

    def __init__(self, c_in=3, width=64, depth=10):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        layers = [nn.Conv2d(c_in, width, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, 1, 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(width, c_in, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        pred_noise = self.net(r)
        return r - pred_noise
