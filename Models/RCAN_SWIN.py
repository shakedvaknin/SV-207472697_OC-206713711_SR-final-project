import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

# -------------------------
# Channel Attention Layer (SE block)
# -------------------------
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
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
        return self.body(x) + x

# -------------------------
# Residual Group
# -------------------------
class ResidualGroup(nn.Module):
    def __init__(self, channel, n_RCAB):
        super(ResidualGroup, self).__init__()
        layers = [RCAB(channel) for _ in range(n_RCAB)]
        layers.append(nn.Conv2d(channel, channel, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + x

# -------------------------
# Swin Transformer Feature Enhancer
# -------------------------
class SwinFeatureEnhancer(nn.Module):
    def __init__(self, channel, input_resolution=(128, 128), num_heads=4, window_size=8):
        super().__init__()
        self.input_resolution = input_resolution
        self.swin_block = SwinTransformerBlock(
            dim=channel,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            attn_drop=0.0,
            drop_path=0.1,
            norm_layer=nn.LayerNorm
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.input_resolution

        # Convert to channel-last: [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # Run SwinTransformerBlock (expects [B, H, W, C])
        x = self.swin_block(x)

        # Back to channel-first: [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
# -------------------------
# RCAN + Swin Model
# -------------------------
class RCAN_Swin(nn.Module):
    def __init__(self, scale=4, num_channels=3, n_resgroups=10, n_RCAB=20, channel=64,
                swin_resolution=(128, 128), swin_heads=4):
        super(RCAN_Swin, self).__init__()
        self.head = nn.Conv2d(num_channels, channel, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(channel, n_RCAB) for _ in range(n_resgroups)],
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        self.swin = SwinFeatureEnhancer(channel, input_resolution=swin_resolution, num_heads=swin_heads)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel, channel * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channel, num_channels, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        res = self.swin(res)  # Output: [B, C, H, W]
        x = self.upsample(res)
        return x
