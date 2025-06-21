import torch
import torch.nn as nn

# -------------------------
# Channel Attention Layer
# -------------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# -------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------
class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x

# -------------------------
# Residual Group
# -------------------------
class ResidualGroup(nn.Module):
    def __init__(self, channel, n_RCAB):
        super(ResidualGroup, self).__init__()
        modules = [RCAB(channel) for _ in range(n_RCAB)]
        modules.append(nn.Conv2d(channel, channel, 3, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        return res + x

# -------------------------
# RCAN Main Model
# -------------------------
class RCAN(nn.Module):
    def __init__(self, scale=2, num_channels=3, n_resgroups=10, n_RCAB=20, channel=64):
        super(RCAN, self).__init__()
        self.head = nn.Conv2d(num_channels, channel, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(channel, n_RCAB) for _ in range(n_resgroups)],
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(channel, channel * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channel, num_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.upsample(res)
        return x
