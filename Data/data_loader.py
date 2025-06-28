import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

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
