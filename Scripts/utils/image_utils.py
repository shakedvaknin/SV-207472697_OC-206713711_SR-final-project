import os
from PIL import Image

def resize_lr_images(folder_path, target_size=(512, 512)):
    """
    Iterates over a folder and resizes all image files containing '_lr' in their filename to 512x512.
    
    Args:
        folder_path (str): Path to the folder with images.
        target_size (tuple): Target size to resize images to. Default is (512, 512).
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    for filename in os.listdir(folder_path):
        if "_lr" in filename and filename.lower().endswith(valid_exts):
            file_path = os.path.join(folder_path, filename)

            try:
                with Image.open(file_path) as img:
                    img_resized = img.resize(target_size, Image.BICUBIC)
                    img_resized.save(file_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
