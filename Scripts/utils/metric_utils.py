import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, data_range=1.0) -> float:
    """
    Compute average PSNR between super-resolved and high-res images.
    Args:
        sr: Super-resolved image tensor [B, C, H, W]
        hr: High-res ground truth tensor [B, C, H, W]
    Returns:
        PSNR value averaged over the batch
    """
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    psnr = 0.0
    for i in range(sr_np.shape[0]):
        psnr += peak_signal_noise_ratio(hr_np[i].transpose(1, 2, 0), 
                                        sr_np[i].transpose(1, 2, 0), 
                                        data_range=data_range)
    return psnr / sr_np.shape[0]

def compute_ssim_batch(sr: torch.Tensor, hr: torch.Tensor, data_range=1.0) -> float:
    """
    Compute average SSIM over a batch.
    Args:
        sr: Super-resolved tensor [B, C, H, W]
        hr: High-res ground truth tensor [B, C, H, W]
    Returns:
        Average SSIM value over batch
    """
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    ssim = 0.0
    for i in range(sr_np.shape[0]):
        ssim += structural_similarity(hr_np[i].transpose(1, 2, 0),
                                    sr_np[i].transpose(1, 2, 0),
                                    channel_axis=2,
                                    data_range=data_range)
    return ssim / sr_np.shape[0]
