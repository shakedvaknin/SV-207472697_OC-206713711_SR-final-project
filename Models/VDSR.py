import torch.nn as nn

class VDSR(nn.Module):
    def __init__(self, num_channels=1):
        super(VDSR, self).__init__()
        layers = [nn.Conv2d(num_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + x  # Residual learning


