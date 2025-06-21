# === Standard Library ===
import json
import os
import random
import tempfile
from pathlib import Path

# === Third-Party Libraries ===
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

# === Local Modules ===
from Scripts.utils.metric_utils import compute_psnr, compute_ssim_batch
from Scripts.utils.plot_utils import annotate_image, create_collage

def convert_to_builtin(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def train_val_test(model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        loss_fn=nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        lr=1e-4,
        num_epochs=20,
        device=None,
        save_dir=None,
        verbose=True,
        val_fid_interval=5):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    best_val_psnr = 0
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_fid': []}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            output = model(lr_img)
            output = output.clamp(0, 1) # Ensure output is in [0, 1] range
            loss = loss_fn(output, hr_img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                output = model(lr_img)
                psnr = compute_psnr(output, hr_img)
                ssim = compute_ssim_batch(output, hr_img)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        # Optional FID computation on validation
        if (epoch + 1) % val_fid_interval == 0:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            with tempfile.TemporaryDirectory() as tmpdir:
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    with torch.no_grad():
                        sr_img = model(lr_img)
                    sr_resized = F.interpolate(sr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    hr_resized = F.interpolate(hr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    fid_metric.update(sr_resized, real=False)
                    fid_metric.update(hr_resized, real=True)
                val_fid = fid_metric.compute().item()
                history['val_fid'].append(val_fid)
                if verbose:
                    print(f"Val FID (epoch {epoch+1}): {val_fid:.4f}")
        else:
            history['val_fid'].append(None)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

        if save_dir and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # Final test evaluation including FID and selective image saving
    model.eval()
    psnr_list, ssim_list = [], []

    # Save only N examples
    N = 10
    example_indices = random.sample(range(len(test_loader.dataset)), N) if len(test_loader.dataset) >= N else list(range(len(test_loader.dataset)))
    saved_idx = 0

    collage_dir = Path(save_dir) / "collages"
    collage_dir.mkdir(parents=True, exist_ok=True)

    temp_out = Path(tempfile.mkdtemp()) / "output"
    temp_gt = Path(tempfile.mkdtemp()) / "gt"
    temp_out.mkdir(parents=True, exist_ok=True)
    temp_gt.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (lr_img, hr_img) in enumerate(test_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            output = model(lr_img)
            psnr = compute_psnr(output, hr_img)
            ssim = compute_ssim_batch(output, hr_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            save_image(output.clamp(0, 1), temp_out / f"{idx:05d}.png")
            save_image(hr_img.clamp(0, 1), temp_gt / f"{idx:05d}.png")

            if idx in example_indices:
                caption = f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
                lr_annot = annotate_image(lr_img, caption)
                sr_annot = annotate_image(output, caption)
                hr_annot = annotate_image(hr_img, caption)

                collage = [TF.to_pil_image(t.squeeze(0).cpu()) for t in [lr_annot, sr_annot, hr_annot]]
                collage_path = collage_dir / f"{saved_idx:05d}_PSNR{psnr:.2f}_SSIM{ssim:.4f}.png"
                create_collage(collage, collage_path)

                saved_idx += 1

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    with torch.no_grad():
        for fname in sorted(os.listdir(temp_out)):
            sr_img = Image.open(temp_out / fname).convert('RGB')
            hr_img = Image.open(temp_gt / fname).convert('RGB')

            sr_tensor = TF.to_tensor(sr_img).unsqueeze(0).to(device)
            hr_tensor = TF.to_tensor(hr_img).unsqueeze(0).to(device)

            sr_tensor = F.interpolate(sr_tensor, size=(299, 299), mode='bilinear', align_corners=False)
            hr_tensor = F.interpolate(hr_tensor, size=(299, 299), mode='bilinear', align_corners=False)

            fid_metric.update(sr_tensor, real=False)
            fid_metric.update(hr_tensor, real=True)

    fid_score = fid_metric.compute().item()

    final_metrics = {
        'test_psnr': float(np.mean(psnr_list)),
        'test_ssim': float(np.mean(ssim_list)),
        'test_fid': float(fid_score)
    }

    if verbose:
        print("\n=== Final Test Metrics ===")
        print("PSNR:", final_metrics['test_psnr'])
        print("SSIM:", final_metrics['test_ssim'])
        print("FID :", final_metrics['test_fid'])

    if save_dir:
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(convert_to_builtin({**history, **final_metrics}), f, indent=2)

    return model, history, final_metrics
