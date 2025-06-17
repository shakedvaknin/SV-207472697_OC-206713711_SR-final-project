from imports import *
from helpers import save_batch_as_images
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize

def train_SvOcSRCNN(
    model,
    train_loader,
    val_loader,
    perceptual_loss_fn,
    num_epochs=50,
    alpha=0.8,
    lr=1e-4,
    device='cuda',
    save_path='svocsrcnn.pth',
):
    model = model.to(device)
    perceptual_loss_fn = perceptual_loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'psnr': [],
        'ssim': [],
        'fid': [],
    }

    best_fid = float("inf")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    resizer = Resize((299, 299))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for lr_imgs, hr_imgs in loop:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss_l1 = F.l1_loss(sr_imgs, hr_imgs)
            loss_perc = perceptual_loss_fn(sr_imgs, hr_imgs)
            loss = alpha * loss_l1 + (1 - alpha) * loss_perc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ---- Evaluation ----
        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                sr_imgs = model(lr_imgs)
                loss_l1 = F.l1_loss(sr_imgs, hr_imgs)
                loss_perc = perceptual_loss_fn(sr_imgs, hr_imgs)
                loss = alpha * loss_l1 + (1 - alpha) * loss_perc
                val_loss += loss.item()

                # FID update
                fid_metric.update(resizer(sr_imgs), real=False)
                fid_metric.update(resizer(hr_imgs), real=True)

                # Metrics for first image
                sr_np = sr_imgs[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                hr_np = hr_imgs[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()

                psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
                ssim = compare_ssim(hr_np, sr_np, channel_axis=-1, data_range=1.0)

                psnr_total += psnr
                ssim_total += ssim

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_total / len(val_loader)
        avg_ssim = ssim_total / len(val_loader)

        fid_score = fid_metric.compute().item()
        fid_metric.reset()

        history['val_loss'].append(avg_val_loss)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        history['fid'].append(fid_score)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | FID: {fid_score:.2f}")

        # Save model with lowest FID
        if fid_score < best_fid:
            best_fid = fid_score
            torch.save(model.state_dict(), save_path)

    return history
