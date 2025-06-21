import os
import zipfile
import uuid
import requests
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def download_div2k(destination="data"):
    """
    Downloads the DIV2K dataset zip file, extracts it to 'data/DIV2K',
    flattens all subfolders (copies images to a single folder), and removes duplicates.
    """
    os.makedirs(destination, exist_ok=True)
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = os.path.join(destination, "DIV2K_train_HR.zip")
    extract_temp = os.path.join(destination, "_temp_extract")
    final_folder = os.path.join(destination, "DIV2K")

    # === Download ZIP ===
    if not os.path.exists(zip_path):
        print("Downloading DIV2K...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
    else:
        print("DIV2K zip already exists.")

    # === Extract ZIP to temp directory ===
    if not os.path.exists(final_folder):
        print("Extracting DIV2K...")
        os.makedirs(extract_temp, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        print("Extraction complete.")

        # Flatten and copy all images to DIV2K directory, remove duplicates
        os.makedirs(final_folder, exist_ok=True)
        seen = set()
        for root, _, files in os.walk(extract_temp):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(final_folder, file)
                    if file not in seen:
                        os.rename(src_path, dst_path)
                        seen.add(file)
        # Clean up temporary extraction
        for root, dirs, files in os.walk(extract_temp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extract_temp)
        print("Files flattened and duplicates removed.")

def preprocess_div2k(source_folder="Data/DIV2K", target_folder="Data/DIV2K_NORMALIZED", standard_size=(2048, 1408)):
    """
    Resizes all images in 'source_folder' to a fixed standard resolution (default 2048x1408)
    and saves them into 'target_folder'. Uses img.load() to catch any corrupted files.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    print(f"Preprocessing images to standard resolution: {standard_size}...")

    for img_name in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_name)
        target_path = os.path.join(target_folder, img_name)

        try:
            with ImageFile.open(img_path) as img:
                img.load()  # 🔹 Ensure full image is read to trigger exception if corrupted
                img = img.resize(standard_size, ImageFile.BICUBIC)
                img.save(target_path)
        except Exception as e:
            print(f"Skipping {img_name}: {e}")

    print("Preprocessing complete. Normalized images saved to:", target_folder)

    def save_batch_as_images(batch_tensor, root_dir):
        """
        Saves a batch of tensors (images) to the specified directory as PNG files.
        Each tensor in the batch is saved with a unique UUID filename.
        """
        os.makedirs(root_dir, exist_ok=True)
        for i, img in enumerate(batch_tensor):
            img = img.clamp(0, 1).cpu()
            vutils.save_image(img, os.path.join(root_dir, f"{uuid.uuid4().hex}.png"))
