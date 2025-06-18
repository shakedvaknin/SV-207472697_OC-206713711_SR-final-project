import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, hr_folder, scale=2):
        self.hr_folder = hr_folder
        self.hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('.png')])
        self.scale = scale
        self.hr_transform = transforms.Compose([
            transforms.Resize((256, 256)),   # Resize all HR images to same size
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),  # Downscale
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),  # Upscale
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_image = Image.open(os.path.join(self.hr_folder, self.hr_files[idx])).convert('RGB')
        hr_tensor = self.hr_transform(hr_image)
        lr_tensor = self.lr_transform(hr_image)
        return lr_tensor, hr_tensor