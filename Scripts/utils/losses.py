from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from piq import ssim

class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        # Load pretrained VGG19 and use layers up to relu2_2 (layer 8)
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:9].to(device).eval()

        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_extractor = vgg
        self.criterion = nn.MSELoss()

        # Normalize to match VGG19 input expectations
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, sr, hr):
        """
        sr, hr: [B, 3, H, W] images in [0, 1] range
        """
        # Normalize each image in batch
        sr_norm = torch.stack([self.normalize(img) for img in sr])
        hr_norm = torch.stack([self.normalize(img) for img in hr])

        sr_feat = self.feature_extractor(sr_norm)
        hr_feat = self.feature_extractor(hr_norm)

        return self.criterion(sr_feat, hr_feat)

class NewCombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.4, data_range=1.0):
        """
        alpha: weight for MSE
        beta: weight for L1
        (1 - alpha - beta): weight for SSIM
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.data_range = data_range
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        mse_loss = self.mse(prediction, target)
        l1_loss = self.l1(prediction, target)
        prediction = prediction.clamp(0,1)
        target = target.clamp(0,1)
        ssim_val = ssim(prediction, target, data_range=self.data_range)
        ssim_loss = 1 - ssim_val  # Higher SSIM = better, so loss = 1 - SSIM
        return self.alpha * mse_loss + self.beta * l1_loss + (1 - self.alpha - self.beta) * ssim_loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, resize=True, device='cpu'):
        super().__init__()
        self.perceptual = PerceptualLoss(device=device)
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, sr, hr):
        #l1 = F.l1_loss(sr, hr)
        l2 = self.mse(sr, hr)
        perceptual = self.perceptual(sr, hr)
        return self.alpha * l2 + (1 - self.alpha) * perceptual

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))