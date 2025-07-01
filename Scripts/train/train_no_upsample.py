import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import os
from pathlib import Path
import json
import random
import tempfile
from PIL import Image
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from Scripts.utils.image_utils import resize_lr_images
from Scripts.utils.metric_utils import compute_psnr, compute_ssim_batch
from Scripts.utils.plot_utils import create_collage

def train_and_validate_no_upsample(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    save_dir,
    checkpoint_dir="checkpoints",
    model_name="RCAN",
    num_epochs=20,
    val_fid_interval=5,
    device=None,
    verbose=True,
    early_stopping_patience=10,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_psnr = 0
    best_epoch = -1
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_fid': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            output = model(lr_img).clamp(0, 1)
            loss = loss_fn(output, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                output = model(lr_img).clamp(0, 1)
                val_loss_total += loss_fn(output, hr_img).item()
                psnr_list.append(compute_psnr(output, hr_img))
                ssim_list.append(compute_ssim_batch(output, hr_img))

        avg_val_loss = val_loss_total / len(val_loader)
        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)

        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        if (epoch + 1) % val_fid_interval == 0:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            with tempfile.TemporaryDirectory() as tmpdir:
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    sr_img = model(lr_img).clamp(0, 1)
                    sr_resized = F.interpolate(sr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    hr_resized = F.interpolate(hr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    fid_metric.update(sr_resized, real=False)
                    fid_metric.update(hr_resized, real=True)
                history['val_fid'].append(fid_metric.compute().item())
        else:
            history['val_fid'].append(None)

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}")

        if checkpoint_dir and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, model_name + '_best_model.pth'))
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. No improvement in PSNR for {early_stopping_patience} epochs.")
            break

    # === Save full training history ===
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def test_no_upsample(
    model,
    test_loader,
    save_dir,
    forced_indices=None,
    model_name="RCAN",
    checkpoint_dir="checkpoints",
    device=None,
    verbose=True
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_dir:
        best_model_path = os.path.join(checkpoint_dir, model_name + '_best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            if verbose:
                print(f"Loaded best model from: {best_model_path}")

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
            output = model(lr_img).clamp(0, 1)
            psnr = compute_psnr(output, hr_img)
            ssim = compute_ssim_batch(output, hr_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            collage = [TF.to_pil_image(t.squeeze(0).cpu()) for t in [lr_img, output, hr_img]]
            collage_path = collage_dir / f"{idx:05d}_PSNR{psnr:.2f}_SSIM{ssim:.4f}.png"
            resize_lr_images(example_img_dir, target_size=(512, 512))
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

    final_metrics = {
        'test_psnr': float(np.mean(psnr_list)),
        'test_ssim': float(np.mean(ssim_list)),
        'test_fid': float(fid_metric.compute().item())
    }

    if verbose:
        print("\n=== Final Test Metrics ===")
        print("PSNR:", final_metrics['test_psnr'])
        print("SSIM:", final_metrics['test_ssim'])
        print("FID :", final_metrics['test_fid'])

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    with open(os.path.join(save_dir, 'test_examples.json'), 'w') as f:
        json.dump(example_data, f, indent=2)

    return final_metrics, example_data
