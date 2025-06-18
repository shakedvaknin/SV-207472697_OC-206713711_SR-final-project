from PIL import Image, ImageDraw, ImageFont       # For drawing text and making collages
import torchvision.transforms.functional as TF   # For tensor ↔ image conversion
import matplotlib.pyplot as plt
import numpy as np

def annotate_image(tensor_img, text):
    img = TF.to_pil_image(tensor_img.squeeze(0).cpu())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((5, 5), text, fill="white", font=font)
    return TF.to_tensor(img)

def create_collage(images, save_path):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    collage.save(save_path)


# === Training Curves Plot ===
def plot_training_curves(history, save_path=None):

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['val_psnr'], label='Val PSNR')
    plt.plot(epochs, history['val_ssim'], label='Val SSIM')
    if any(history.get('val_fid', [])):
        plt.plot(epochs, [fid if fid is not None else np.nan for fid in history['val_fid']], label='Val FID')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title("Training Curves")
    if save_path:
        plt.savefig(save_path)
    plt.show()
