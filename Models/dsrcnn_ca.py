import torch.nn as nn

# ---------------------- Channel Attention Layer ----------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# ---------------------- Pyramid Deep SRCNN with Channel Attention ----------------------
class PyramidDeepSRCNN_CA(nn.Module):
    def __init__(self, num_channels=3):
        super(PyramidDeepSRCNN_CA, self).__init__()

        # Entry layer
        self.entry = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Progressive channel expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Deep middle blocks (64 channels)
        self.middle_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ) for _ in range(6)
            ],
            CALayer(64)
        )

        # Progressive channel reduction
        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.exit = nn.Conv2d(16, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.middle_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.exit(x)
        return x