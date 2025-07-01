import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from Data.data_loader import DIV2KDataset, TiledDIV2KDataset
from Data.data_utils import download_div2k, add_augmentation

def preprocess_pipeline(config):
    print("Downloading and preparing DIV2K dataset...")
    download_div2k("Data")

    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Apply augmentation if requested
    if config["aug"]:
        add_augmentation("Data/DIV2K")

    # Load dataset
    if config["tiled"]:
        print("Using TiledDIV2KDataset")
        dataset = TiledDIV2KDataset("Data/DIV2K", scale=config["scale"])
    else:
        print("Using DIV2KDataset")
        dataset = DIV2KDataset("Data/DIV2K", scale=config["scale"])

    # Ensure deterministic splitting
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Set shuffle=False to maintain consistent index-to-image mapping
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    forced_indices = config["idx"]

    print("Dataset ready and reproducible.")
    return train_loader, val_loader, test_loader, forced_indices
