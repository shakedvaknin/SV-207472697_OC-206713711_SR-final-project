import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import os

from Models.VDSR import VDSR
from Models.RCAN import RCAN
from Models.FusionNet import FusionNet
from Scripts.losses import CombinedLoss

def train_fusion_net(
    train_loader,
    val_loader,
    vdsr_ckpt="checkpoints/VDSR/best_model.pth",
    rcan_ckpt="checkpoints/RCAN/best_model.pth",
    fusion_ckpt_dir="checkpoints/FusionNet",
    scale=2,
    num_channels=3,
    num_epochs=30,
    lr=1e-4,
    alpha=0.8,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(fusion_ckpt_dir, exist_ok=True)

    # Load pretrained VDSR and RCAN
    vdsr = VDSR(num_channels=num_channels).to(device)
    vdsr.load_state_dict(torch.load(vdsr_ckpt, map_location=device))
    vdsr.eval()

    rcan = RCAN(scale=scale, num_channels=num_channels).to(device)
    rcan.load_state_dict(torch.load(rcan_ckpt, map_location=device))
    rcan.eval()

    # Initialize FusionNet
    fusion_net = FusionNet(in_channels=6).to(device)
    optimizer = torch.optim.Adam(fusion_net.parameters(), lr=lr)
    loss_fn = CombinedLoss(alpha=alpha, device=device)

    # Training loop
    for epoch in range(num_epochs):
        fusion_net.train()
        running_loss = 0.0

        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            lr_up = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)

            with torch.no_grad():
                out_vdsr = vdsr(lr_up).clamp(0, 1)
                out_rcan = rcan(lr_img).clamp(0, 1)

            fusion_input = torch.cat([out_vdsr, out_rcan], dim=1)
            mask = fusion_net(fusion_input).sigmoid()
            out_fused = (1 - mask) * out_vdsr + mask * out_rcan

            loss = loss_fn(out_fused, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        torch.save(fusion_net.state_dict(), os.path.join(fusion_ckpt_dir, "best_model.pth"))
