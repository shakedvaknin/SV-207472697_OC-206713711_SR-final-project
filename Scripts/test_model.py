from imports import *

def evaluate_and_collect(model, name, device, test_loader):
    psnr_list = []
    ssim_list = []

    with tempfile.TemporaryDirectory() as tmpdir:
        gen_dir = os.path.join(tmpdir, "gen")
        real_dir = os.path.join(tmpdir, "real")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        for i, (lr, hr) in enumerate(test_loader):
            lr, hr = lr.to(device), hr.to(device)

            with torch.no_grad():
                sr = model(lr)

            sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
            hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)

            psnr_val = psnr_metric(hr_np, sr_np, data_range=1.0)
            ssim_val = ssim_metric(hr_np, sr_np, channel_axis=2, data_range=1.0)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            sr_img = (sr_np * 255).clip(0, 255).astype(np.uint8)
            hr_img = (hr_np * 255).clip(0, 255).astype(np.uint8)

            Image.fromarray(sr_img).save(os.path.join(gen_dir, f"{i:04d}.png"))
            Image.fromarray(hr_img).save(os.path.join(real_dir, f"{i:04d}.png"))

        fid_score = fid.compute_fid(gen_dir, real_dir)

    return {
        "PSNR": round(np.mean(psnr_list), 4),
        "SSIM": round(np.mean(ssim_list), 4),
        "FID": round(fid_score, 4)
    }