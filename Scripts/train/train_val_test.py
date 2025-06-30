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

from Scripts.utils.image_utils import resize_lr_images
from Scripts.utils.metric_utils import compute_psnr, compute_ssim_batch
from Scripts.utils.plot_utils import create_collage

def train_and_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    save_dir,
    checkpoint_dir="checkpoints",
    model_name="Model",
    num_epochs=20,
    val_fid_interval=5,
    device=None,
    verbose=True,
    early_stopping_patience=10,
    use_wandb=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_psnr = 0
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_fid': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for lr_img, hr_img in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
            output = model(lr_up).clamp(0, 1)
            loss = loss_fn(output, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        history['train_loss'].append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
                output = model(lr_up).clamp(0, 1)

                val_loss_total += loss_fn(output, hr_img).item()
                psnr_list.append(compute_psnr(output, hr_img))
                ssim_list.append(compute_ssim_batch(output, hr_img))

        avg_val_loss = val_loss_total / len(val_loader)
        val_psnr = np.mean(psnr_list)
        val_ssim = np.mean(ssim_list)

        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        # === Optional FID ===
        if (epoch + 1) % val_fid_interval == 0:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                    lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
                    output = model(lr_up).clamp(0, 1)
                    sr_resized = F.interpolate(output, size=(299, 299), mode='bilinear', align_corners=False)
                    hr_resized = F.interpolate(hr_img, size=(299, 299), mode='bilinear', align_corners=False)
                    fid_metric.update(sr_resized, real=False)
                    fid_metric.update(hr_resized, real=True)
                fid_score = fid_metric.compute().item()
                history['val_fid'].append(fid_score)
        else:
            fid_score = None
            history['val_fid'].append(None)

        # === Logging ===
        if verbose:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                  f"PSNR={val_psnr:.2f}, SSIM={val_ssim:.4f}, FID={fid_score}")

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_fid": fid_score
            })

        # === Checkpointing ===
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, model_name + '_best_model.pth'))
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}. No improvement in PSNR for {early_stopping_patience} epochs.")
            break

    # === Append training history to metrics.json ===
    metrics_path = os.path.join(save_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing_data = json.load(f)
        if isinstance(existing_data, list):
            existing_data.append(history)
        else:
            existing_data = [existing_data, history]
    else:
        existing_data = [history]

    with open(metrics_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    return model, history


def test_upsample(
    model,
    test_loader,
    save_dir,
    checkpoint_dir="checkpoints",
    model_name="Model",
    forced_indices=None,
    device=None,
    verbose=True,
    use_wandb=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_path = os.path.join(checkpoint_dir, model_name + '_best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        if verbose:
            print(f"Loaded best model from checkpoint: {best_model_path}")

    model.eval()
    psnr_list, ssim_list = [], []
    collage_dir = Path(save_dir) / "collages"
    example_dir = Path(save_dir) / "test_examples"
    os.makedirs(collage_dir, exist_ok=True)
    os.makedirs(example_dir, exist_ok=True)

    example_data = {}
    dataset = test_loader.dataset

    if forced_indices is None:
        forced_indices = sorted(random.sample(range(len(dataset)), 10))

    with torch.no_grad():
        for idx in forced_indices:
            lr_img, hr_img = dataset[idx]
            lr_img, hr_img = lr_img.unsqueeze(0).to(device), hr_img.unsqueeze(0).to(device)
            lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
            output = model(lr_up).clamp(0, 1)

            psnr = compute_psnr(output, hr_img)
            ssim = compute_ssim_batch(output, hr_img)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            collage = [TF.to_pil_image(t.squeeze(0).cpu()) for t in [lr_img, output, hr_img]]
            collage_path = collage_dir / f"{idx:05d}_PSNR{psnr:.2f}_SSIM{ssim:.4f}.png"
            resize_lr_images(example_dir, target_size=(512, 512))
            create_collage(collage, collage_path)

            paths = {
                "lr": example_dir / f"{idx}_lr.png",
                "sr": example_dir / f"{idx}_sr.png",
                "hr": example_dir / f"{idx}_hr.png"
            }

            save_image(lr_img.clamp(0, 1), paths["lr"])
            save_image(output.clamp(0, 1), paths["sr"])
            save_image(hr_img.clamp(0, 1), paths["hr"])

            example_data[idx] = {
                "lr": str(paths["lr"]),
                "sr": str(paths["sr"]),
                "hr": str(paths["hr"]),
                "psnr": float(psnr),
                "ssim": float(ssim)
            }

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    with torch.no_grad():
        for entry in example_data.values():
            sr = TF.to_tensor(Image.open(entry["sr"]).convert("RGB")).unsqueeze(0).to(device)
            hr = TF.to_tensor(Image.open(entry["hr"]).convert("RGB")).unsqueeze(0).to(device)
            sr = F.interpolate(sr, size=(299, 299), mode='bilinear', align_corners=False)
            hr = F.interpolate(hr, size=(299, 299), mode='bilinear', align_corners=False)
            fid_metric.update(sr, real=False)
            fid_metric.update(hr, real=True)

    final_metrics = {
        "test_psnr": float(np.mean(psnr_list)),
        "test_ssim": float(np.mean(ssim_list)),
        "test_fid": float(fid_metric.compute().item())
    }

    if verbose:
        print("\n=== Final Test Metrics ===")
        for k, v in final_metrics.items():
            print(f"{k.upper()}: {v:.4f}")

    with open(Path(save_dir) / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    with open(Path(save_dir) / "test_examples.json", 'w') as f:
        json.dump(example_data, f, indent=2)

    if use_wandb:
        wandb.log(final_metrics)
        for idx in list(example_data.keys())[:3]:  # log 3 example images
            wandb.log({
                f"Example_{idx}": [
                    wandb.Image(str(example_data[idx]["lr"]), caption="LR"),
                    wandb.Image(str(example_data[idx]["sr"]), caption="SR"),
                    wandb.Image(str(example_data[idx]["hr"]), caption="HR"),
                ]
            })

    return final_metrics, example_data
