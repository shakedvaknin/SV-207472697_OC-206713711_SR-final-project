import os
import zipfile
import uuid
import requests
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def download_div2k(destination="data"):
    """
    Downloads and extracts DIV2K training, validation, and test HR sets into a single folder.
    Flattens the structure and skips existing files.
    """
    os.makedirs(destination, exist_ok=True)

    # URLs for different splits
    urls = {
        "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        #"test":  "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_test_HR.zip"
    }

    final_folder = os.path.join(destination, "DIV2K")
    os.makedirs(final_folder, exist_ok=True)
    temp_extract = os.path.join(destination, "_temp_extract")
    os.makedirs(temp_extract, exist_ok=True)

    for split, url in urls.items():
        zip_filename = f"DIV2K_{split}_HR.zip"
        zip_path = os.path.join(destination, zip_filename)

        # === Download ZIP if not already downloaded ===
        if not os.path.exists(zip_path):
            print(f"Downloading {zip_filename}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {split}"):
                        f.write(chunk)
        else:
            print(f"{zip_filename} already exists.")

        # === Extract to temp folder ===
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)

        # Move image files into final folder (skip duplicates)
        for root, _, files in os.walk(temp_extract):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(final_folder, file)

                    if not os.path.exists(dst_path):
                        os.rename(src_path, dst_path)  # move only if not exists

        # Clean temp folder
        for root, dirs, files in os.walk(temp_extract, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.rmdir(temp_extract)
    print("All splits extracted and flattened into DIV2K folder (duplicates skipped).")


def add_augmentation(directory, num_threads=8):
    """
    Augments images in the specified directory using multithreading.
    Skips files already augmented. Saves rotated, h-flipped, and v-flipped versions.
    """
    suffixes = ["_rot", "_hflip", "_vflip"]
    valid_exts = (".png", ".jpg", ".jpeg")

    # Filter base images
    files_to_process = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(valid_exts) and not any(suffix in fname for suffix in suffixes):
            files_to_process.append(fname)

    def augment_file(fname):
        base, ext = os.path.splitext(fname)
        path = os.path.join(directory, fname)

        try:
            img = Image.open(path).convert("RGB")

            # Rotate 90°
            img_rot = TF.rotate(img, 90)
            img_rot.save(os.path.join(directory, f"{base}_rot{ext}"))

            # Horizontal flip
            img_hflip = TF.hflip(img)
            img_hflip.save(os.path.join(directory, f"{base}_hflip{ext}"))

            # Vertical flip
            img_vflip = TF.vflip(img)
            img_vflip.save(os.path.join(directory, f"{base}_vflip{ext}"))

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    # Run in parallel
    print(f"Starting augmentation on {len(files_to_process)} images using {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(augment_file, files_to_process), total=len(files_to_process)))

    print("Augmentation complete.")

def preprocess_div2k_center_crop(source_folder="Data/DIV2K", target_folder="Data/DIV2K_CROPPED", patch_size=(512, 512)):
    """
    Extracts a centered patch of size `patch_size` (default 512x512) from each image in 'source_folder'
    and saves the patch to 'target_folder'. Skips corrupted or incompatible images.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    print(f"Cropping center patches of size {patch_size}...")

    for img_name in os.listdir(source_folder):
        img_path = os.path.join(source_folder, img_name)
        target_path = os.path.join(target_folder, img_name)

        try:
            with Image.open(img_path) as img:
                img.load()  # Ensure full image is read

                width, height = img.size
                crop_width, crop_height = patch_size

                if width < crop_width or height < crop_height:
                    print(f"Skipping {img_name}: image too small ({width}x{height})")
                    continue

                left = (width - crop_width) // 2
                upper = (height - crop_height) // 2
                right = left + crop_width
                lower = upper + crop_height

                cropped_img = img.crop((left, upper, right, lower))
                cropped_img.save(target_path)

        except Exception as e:
            print(f"Skipping {img_name}: {e}")

    print("Center cropping complete. Cropped patches saved to:", target_folder)


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
            with Image.open(img_path) as img:
                img.load()  # 🔹 Ensure full image is read to trigger exception if corrupted
                img = img.resize(standard_size, Image.BICUBIC)
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
