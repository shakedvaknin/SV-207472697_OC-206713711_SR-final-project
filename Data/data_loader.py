import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class DIV2KDataset(Dataset):
    def __init__(self, hr_folder = "Data/DIV2K", scale=4, crop_size=512):
        self.hr_folder = hr_folder
        self.hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('.png')])
        self.scale = scale
        self.crop_size = crop_size
        self.lr_size = crop_size // scale

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_folder, self.hr_files[idx])
        hr_image = Image.open(hr_path).convert('RGB')

        # Random crop
        w, h = hr_image.size
        if w < self.crop_size or h < self.crop_size:
            raise ValueError(f"Image too small: {hr_path}")

        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        hr_crop = hr_image.crop((x, y, x + self.crop_size, y + self.crop_size))

        # Downsample to get LR
        lr_crop = hr_crop.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # Transform to tensors
        transform = transforms.ToTensor()
        return transform(lr_crop), transform(hr_crop)
    


class TiledDIV2KDataset(Dataset):
    def __init__(self, hr_folder="Data/DIV2K", scale=4, crop_size=512, num_threads=8):
        self.hr_folder = hr_folder
        self.hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('.png')])
        self.scale = scale
        self.crop_size = crop_size
        self.lr_size = crop_size // scale
        self.transform = transforms.ToTensor()
        self.samples = []

        print(f"Preprocessing {len(self.hr_files)} HR images for tiling using {num_threads} threads...")

        def process_image(file_idx_file):
            file_idx, file_name = file_idx_file
            img_path = os.path.join(self.hr_folder, file_name)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                tiles_w = w // self.crop_size
                tiles_h = h // self.crop_size
                return [(file_idx, j * self.crop_size, i * self.crop_size)
                        for i in range(tiles_h) for j in range(tiles_w)]
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                return []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(
                executor.map(process_image, enumerate(self.hr_files)),
                total=len(self.hr_files),
                desc="Tiling DIV2K"
            ))

        # Flatten list of lists
        self.samples = [sample for sublist in results for sample in sublist]

        print(f"Total crop pairs generated: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, x, y = self.samples[idx]
        file_name = self.hr_files[file_idx]
        img_path = os.path.join(self.hr_folder, file_name)

        with Image.open(img_path) as img:
            hr_image = img.convert("RGB")
            hr_crop = hr_image.crop((x, y, x + self.crop_size, y + self.crop_size))
            lr_crop = hr_crop.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        return self.transform(lr_crop), self.transform(hr_crop)