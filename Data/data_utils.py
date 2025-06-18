import os
import zipfile
import uuid
import requests
from tqdm import tqdm
import torchvision.utils as vutils

def download_div2k(destination="data"):
    os.makedirs(destination, exist_ok=True)
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    filename = os.path.join(destination, "DIV2K_train_HR.zip")

    if not os.path.exists(filename):
        print("Downloading DIV2K...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
    else:
        print("DIV2K already downloaded.")

    # Extract
    extract_path = os.path.join(destination, "DIV2K_train_HR")
    if not os.path.exists(extract_path):
        print("Extracting DIV2K...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print("Done.")
    else:
        print("DIV2K already extracted.")

def save_batch_as_images(batch_tensor, root_dir):
    """Save a batch of images to the specified directory."""
    for i, img in enumerate(batch_tensor):
        img = img.clamp(0, 1).cpu()
        vutils.save_image(img, os.path.join(root_dir, f"{uuid.uuid4().hex}.png"))

