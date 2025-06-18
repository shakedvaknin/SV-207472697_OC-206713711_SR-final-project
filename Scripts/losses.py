from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True, device='cpu'):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
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
        f1 = self.vgg(sr)
        f2 = self.vgg(hr)
        return F.l1_loss(f1, f2)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, resize=True, device='cpu'):
        super().__init__()
        self.perceptual = PerceptualLoss(resize=resize, device=device)
        self.alpha = alpha

    def forward(self, sr, hr):
        l1 = F.l1_loss(sr, hr)
        perceptual = self.perceptual(sr, hr)
        return self.alpha * l1 + (1 - self.alpha) * perceptual

