import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont       # For drawing text and making collages
import torchvision.transforms.functional as TF   # For tensor ↔ image conversion
import matplotlib.pyplot as plt
import numpy as np
import os


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

def generate_summary_collage_from_checkpoints(checkpoints_root="checkpoints", output_dir="checkpoints/summary_collages"):
    model_dirs = [d for d in os.listdir(checkpoints_root) if os.path.isdir(os.path.join(checkpoints_root, d)) and d != "summary_collages"]

    all_model_outputs = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for model_name in model_dirs:
        json_path = os.path.join(checkpoints_root, model_name, "test_examples.json")
        if not os.path.exists(json_path):
            print(f"Skipping {model_name}, no test_examples.json found.")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        all_model_outputs[model_name] = {}
        for idx_str, entry in data.items():
            idx = int(idx_str)
            all_model_outputs[model_name][idx] = {
                "lr": TF.to_tensor(Image.open(entry["lr"]).convert("RGB")).unsqueeze(0),
                "sr": TF.to_tensor(Image.open(entry["sr"]).convert("RGB")).unsqueeze(0),
                "hr": TF.to_tensor(Image.open(entry["hr"]).convert("RGB")).unsqueeze(0),
                "psnr": entry["psnr"],
                "ssim": entry["ssim"]
            }
    # Create the collage
    create_multi_model_collage()


def create_multi_model_collage(root_dir: str = "checkpoints", font_path: str = None, font_size: int = 22):
    """
    Generates a comparison collage of model outputs from multiple models.
    Each row represents an example index, and each column represents a model's SR image with LR and HR as the first two columns.
    """
    root = Path(root_dir)
    model_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "test_examples.json").exists() and (d / "metrics.json").exists()]
    model_dirs = sorted(model_dirs, key=lambda d: d.name)

    if not model_dirs:
        raise ValueError("No valid model directories found.")

    # Load test examples and ensure consistent example indices
    model_data = {}
    example_indices = None
    for model_dir in model_dirs:
        name = model_dir.name
        with open(model_dir / "test_examples.json") as f:
            examples = json.load(f)
        with open(model_dir / "metrics.json") as f:
            metrics = json.load(f)

        indices = sorted(map(int, examples.keys()))
        if example_indices is None:
            example_indices = indices
        elif example_indices != indices:
            raise ValueError(f"Example indices do not match across models. Check model {name}.")

        model_data[name] = {"examples": examples, "metrics": metrics}

    num_examples = len(example_indices)
    num_models = len(model_data)
    columns = ["LR", *model_data.keys(), "HR"]
    cell_width, cell_height = 400, 400 + font_size + 10
    font = ImageFont.truetype(font_path or str(ImageFont.load_default().path), font_size) if font_path else ImageFont.load_default()

    for example_idx in example_indices:
        collage = Image.new("RGB", (len(columns) * cell_width, cell_height), (255, 255, 255))
        draw = ImageDraw.Draw(collage)

        for col_idx, label in enumerate(columns):
            x = col_idx * cell_width
            if label == "LR":
                img_path = model_dirs[0] / "test_examples" / f"{example_idx}_lr.png"
                caption = "Low Resolution"
            elif label == "HR":
                img_path = model_dirs[0] / "test_examples" / f"{example_idx}_hr.png"
                caption = "High Resolution"
            else:
                img_path = Path(model_data[label]["examples"][str(example_idx)]["sr"])
                psnr = model_data[label]["examples"][str(example_idx)]["psnr"]
                ssim = model_data[label]["examples"][str(example_idx)]["ssim"]
                fid = model_data[label]["metrics"]["test_fid"]
                caption = f"{label}\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}, FID: {fid:.2f}"

            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB").resize((cell_width, cell_width), Image.BICUBIC)
            collage.paste(img, (x, 0))
            draw.rectangle([(x, cell_width), (x + cell_width, cell_height)], fill=(255, 255, 255))
            draw.text((x + 5, cell_width + 2), caption, fill=(0, 0, 0), font=font)

        out_path = root /"summary_collages"/ f"{example_idx:03d}_comparison_collage.png"
        collage.save(out_path)
        print(f"Saved collage for example {example_idx} to {out_path}")

    print("All collages created successfully.")