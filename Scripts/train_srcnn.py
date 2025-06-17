from imports import *
from Models.SRCNN_model import SRCNN
from Models.SvOcSRCNN_model import SvOcSRCNN
def train_srcnn(model, dataloader, num_epochs=5, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, optimizer
    if model == "srcnn":
        model = SRCNN().to(device)
    if model == "svocsrcnn":
        model = SvOcSRCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'psnr': [],
        'ssim': [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for lr_imgs, hr_imgs in loop:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        history['train_loss'].append(avg_loss)

        # Evaluate using first image in batch for PSNR/SSIM (quick sanity check)
        model.eval()
        with torch.no_grad():
            val_lr, val_hr = next(iter(dataloader))
            val_lr, val_hr = val_lr.to(device), val_hr.to(device)
            pred_hr = model(val_lr)

            sr_np = pred_hr[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            hr_np = val_hr[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()

            psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
            ssim = compare_ssim(hr_np, sr_np, channel_axis=-1, data_range=1.0)

        history['psnr'].append(psnr)
        history['ssim'].append(ssim)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

    return model, history