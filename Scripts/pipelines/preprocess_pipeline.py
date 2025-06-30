
from Data.data_loader import DIV2KDataset
from Data.data_utils import download_div2k, add_augmentation
from torch.utils.data import DataLoader, random_split
from Data.data_loader import TiledDIV2KDataset

def preprocess_pipeline(config):
    print("Downloading and preparing DIV2K dataset...")
    download_div2k("Data")
    if config["aug"] == True:
        add_augmentation("Data/DIV2K")
    if config["tiled"] == True:
        print("Using TiledDIV2KDataset")
        dataset = TiledDIV2KDataset("Data/DIV2K", scale=config["scale"])
    else:
        print("Using DIV2KDataset")
        dataset = DIV2KDataset("Data/DIV2K", scale=config["scale"])

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    forced_indices = config["idx"]
    
    print("✅ Dataset ready.")
    return train_loader, val_loader, test_loader, forced_indices
