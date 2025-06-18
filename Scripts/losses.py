import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision import transforms


class PerceptualLoss(nn.Module):
    def __init__(self, resize=True, device=None):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = vgg.to(self.device)  # ✅ Move to device
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.resize = resize

    def forward(self, sr, hr):
        sr = self.transform(sr.clamp(0, 1))
        hr = self.transform(hr.clamp(0, 1))

        if self.resize:
            sr = F.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            hr = F.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)

        f1 = self.vgg(sr.to(self.device))
        f2 = self.vgg(hr.to(self.device))
        return F.l1_loss(f1, f2)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, resize=True, device=None):
        super().__init__()
        self.perceptual = PerceptualLoss(resize=resize, device=device)
        self.alpha = alpha

    def forward(self, sr, hr):
        l1 = F.l1_loss(sr, hr)
        perceptual = self.perceptual(sr, hr)
        return self.alpha * l1 + (1 - self.alpha) * perceptual
