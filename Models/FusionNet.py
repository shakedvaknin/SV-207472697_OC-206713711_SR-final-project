import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, in_channels=6):
        super(FusionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # outputs attention mask [B, 1, H, W]
        )

    def forward(self, x):
        return self.net(x)
