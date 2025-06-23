import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from Models.VDSR import VDSR
from Models.RCAN import RCAN
from Models.FusionNet import FusionNet


def attention_fusion(
    lr_img,
    vdsr_model,
    rcan_model,
    fusion_model,
    scale=2,
    device="cpu"
):
    """
    Applies attention-based fusion using outputs from VDSR and RCAN.

    Args:
        lr_img (Tensor): Low-resolution input image [B, C, H, W]
        vdsr_model (nn.Module): Pretrained VDSR model
        rcan_model (nn.Module): Pretrained RCAN model
        fusion_model (nn.Module): Pretrained FusionNet model
        scale (int): Upsampling factor
        device (str or torch.device): Target device for inference

    Returns:
        Tuple[Tensor, Tensor]: Fused image and attention mask
    """
    lr_img = lr_img.to(device)
    lr_up = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)

    with torch.no_grad():
        vdsr_out = vdsr_model(lr_up).clamp(0, 1)
        rcan_out = rcan_model(lr_img).clamp(0, 1)
        fusion_input = torch.cat([vdsr_out, rcan_out], dim=1)
        mask = fusion_model(fusion_input).sigmoid()
        fused_output = (1 - mask) * vdsr_out + mask * rcan_out

    return fused_output, mask

def run_attention_fusion_inference(
    test_loader,
    output_dir="outputs/fusion_results_attention",
    vdsr_ckpt="checkpoints/VDSR_best_model.pth",
    rcan_ckpt="checkpoints/RCAN_best_model.pth",
    fusion_ckpt="checkpoints/FusionNet_best_model.pth",
    scale=2,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    vdsr = VDSR(num_channels=3).to(device)
    vdsr.load_state_dict(torch.load(vdsr_ckpt, map_location=device))
    vdsr.eval()

    rcan = RCAN(scale=scale, num_channels=3).to(device)
    rcan.load_state_dict(torch.load(rcan_ckpt, map_location=device))
    rcan.eval()

    fusion_net = FusionNet(in_channels=6).to(device)
    fusion_net.load_state_dict(torch.load(fusion_ckpt, map_location=device))
    fusion_net.eval()

    # Inference loop
    for idx, (lr_img, _) in enumerate(tqdm(test_loader, desc="Running Attention Fusion Inference")):
        with torch.no_grad():
            out_fused, mask = attention_fusion(
                lr_img,
                vdsr_model=vdsr,
                rcan_model=rcan,
                fusion_model=fusion_net,
                scale=scale,
                device=device
            )

        lr_img = lr_img.to(device)
        lr_up = F.interpolate(lr_img, scale_factor=scale, mode="bicubic", align_corners=False)
        out_vdsr = vdsr(lr_up).clamp(0, 1)
        out_rcan = rcan(lr_img).clamp(0, 1)

        save_image(lr_img, os.path.join(output_dir, f"{idx:04d}_lr.png"))
        save_image(out_vdsr, os.path.join(output_dir, f"{idx:04d}_vdsr.png"))
        save_image(out_rcan, os.path.join(output_dir, f"{idx:04d}_rcan.png"))
        save_image(mask.expand_as(out_rcan), os.path.join(output_dir, f"{idx:04d}_mask.png"))
        save_image(out_fused, os.path.join(output_dir, f"{idx:04d}_fused_attention.png"))

    print("\nAttention-based fusion inference complete. Results saved to:", output_dir)
