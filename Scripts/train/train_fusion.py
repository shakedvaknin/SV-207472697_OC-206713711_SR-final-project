import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import os


def train_fusion_net(
    fusion_net,
    train_loader,
    optimizer,
    loss_fn,
    background_model,
    object_model,
    fusion_ckpt_dir="checkpoints/FusionNet",
    scale=2,
    num_channels=3,
    num_epochs=30,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(fusion_ckpt_dir, exist_ok=True)

    background_model = background_model.to(device)
    background_model.eval()

    object_model = object_model.to(device)
    object_model.eval()

    fusion_net = fusion_net.to(device)

    # Training loop
    for epoch in range(num_epochs):
        fusion_net.train()
        running_loss = 0.0

        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            lr_up = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)

            with torch.no_grad():
                out_background = background_model(lr_up).clamp(0, 1)
                out_object = object_model(lr_img).clamp(0, 1)

            fusion_input = torch.cat([out_background, out_object], dim=1)
            mask = fusion_net(fusion_input).sigmoid()
            out_fused = (1 - mask) * out_background + mask * out_object

            loss = loss_fn(out_fused, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        torch.save(fusion_net.state_dict(), os.path.join(fusion_ckpt_dir, "best_model.pth"))
