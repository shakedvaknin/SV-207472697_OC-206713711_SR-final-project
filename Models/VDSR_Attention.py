import torch
import torch.nn as nn
import torch.nn.functional as F

class VDSR_SA(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_resblocks=18):
        super().__init__()
        self.input_conv = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.resblocks = nn.Sequential(
            *[ResidualBlockSA(num_features) for _ in range(num_resblocks)]
        )
        #self.output_conv = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        self.output_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_channels, 3, padding=1)
        )


    def forward(self, x):
        out = self.relu(self.input_conv(x))
        out = self.resblocks(out)
        out = self.output_conv(out)
        return out + x  # residual learning

class ResidualBlockSA(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sa(out)
        return x + out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        # Compress channels using max-pool and avg-pool and concatenate
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply max-pool and avg-pool along channel axis (dim=1)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        pool = torch.cat([max_pool, avg_pool], dim=1)  # shape (B, 2, H, W)
        attention = self.sigmoid(self.conv(pool))      # shape (B, 1, H, W)
        return x * attention