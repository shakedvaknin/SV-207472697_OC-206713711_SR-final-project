import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, hr_folder, scale=2, hr_size=(256, 256)):
        self.hr_folder = hr_folder
        self.hr_files = sorted([f for f in os.listdir(hr_folder) if f.endswith('.png')])
        self.scale = scale
        self.hr_size = hr_size
        self.lr_size = (hr_size[0] // scale, hr_size[1] // scale)

        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_folder, self.hr_files[idx])
        hr_image = Image.open(hr_path).convert('RGB')

        hr_tensor = self.hr_transform(hr_image)
        lr_tensor = self.lr_transform(hr_image)

        return lr_tensor, hr_tensor
