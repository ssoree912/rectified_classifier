import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, c_in=3, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, width, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(width * 2, 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)
