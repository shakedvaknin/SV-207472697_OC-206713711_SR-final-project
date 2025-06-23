import json
import os
import random
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

import wandb

from Scripts.utils.metric_utils import compute_psnr, compute_ssim_batch
from Scripts.utils.plot_utils import create_collage

def train_val_test(model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        model_name: str = "Model",
        loss_fn=nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        lr=1e-4,
        num_epochs=20,
        device=None,
        save_dir=None,
        verbose=True,
        val_fid_interval=5,
        forced_indices=None,
        use_wandb=False):

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
            upsampled = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
            output = model(upsampled)
            loss = loss_fn(output, hr_img)
            loss.backward()
            optimizer.step()
            output = output.clamp(0, 1)
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                upsampled = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
                output = model(upsampled).clamp(0, 1)
                psnr = compute_psnr(output, hr_img)
                ssim = compute_ssim_batch(output, hr_img)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        if (epoch + 1) % val_fid_interval == 0:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            with tempfile.TemporaryDirectory() as tmpdir:
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    with torch.no_grad():
                        upsampled = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
                        sr_img = model(upsampled).clamp(0, 1)
                    sr_resized = F.interpolate(sr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    hr_resized = F.interpolate(hr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    fid_metric.update(sr_resized, real=False)
                    fid_metric.update(hr_resized, real=True)
                val_fid = fid_metric.compute().item()
                history['val_fid'].append(val_fid)
                if verbose:
                    print(f"Val FID (epoch {epoch+1}): {val_fid:.4f}")
        else:
            val_fid = None
            history['val_fid'].append(None)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_fid": val_fid
            })

        if save_dir and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # === Final Testing ===
    model.eval()
    psnr_list, ssim_list = [], []
    collage_dir = Path(save_dir) / "collages"
    collage_dir.mkdir(parents=True, exist_ok=True)

    example_data = {}
    example_img_dir = Path(save_dir) / "test_examples"
    example_img_dir.mkdir(parents=True, exist_ok=True)

    if forced_indices is None:
        forced_indices = sorted(random.sample(range(len(test_loader.dataset)), 10))

    dataset = test_loader.dataset

    with torch.no_grad():
        for idx in forced_indices:
            lr_img, hr_img = dataset[idx]
            lr_img, hr_img = lr_img.unsqueeze(0).to(device), hr_img.unsqueeze(0).to(device)
            upsampled = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
            output = model(upsampled).clamp(0, 1)
            psnr = compute_psnr(output, hr_img)
            ssim = compute_ssim_batch(output, hr_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            collage = [TF.to_pil_image(t.squeeze(0).cpu()) for t in [lr_img, output, hr_img]]
            collage_path = collage_dir / f"{idx:05d}_PSNR{psnr:.2f}_SSIM{ssim:.4f}.png"
            create_collage(collage, collage_path)

            lr_path = example_img_dir / f"{idx}_lr.png"
            sr_path = example_img_dir / f"{idx}_sr.png"
            hr_path = example_img_dir / f"{idx}_hr.png"
            save_image(lr_img.clamp(0, 1), lr_path)
            save_image(output.clamp(0, 1), sr_path)
            save_image(hr_img.clamp(0, 1), hr_path)

            example_data[idx] = {
                "lr": str(lr_path),
                "sr": str(sr_path),
                "hr": str(hr_path),
                "psnr": float(psnr),
                "ssim": float(ssim)
            }

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    with torch.no_grad():
        for entry in example_data.values():
            sr_tensor = TF.to_tensor(Image.open(entry['sr']).convert("RGB")).unsqueeze(0).to(device)
            hr_tensor = TF.to_tensor(Image.open(entry['hr']).convert("RGB")).unsqueeze(0).to(device)
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

    if use_wandb:
        wandb.log(final_metrics)

    if save_dir:
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump({**history, **final_metrics}, f, indent=2)
        with open(os.path.join(save_dir, 'test_examples.json'), 'w') as f:
            json.dump(example_data, f, indent=2)

    return model, history, final_metrics
