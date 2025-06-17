import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imports import *
from helpers import download_div2k, PerceptualLoss
from Data.data_loader import DIV2KDataset
from Scripts.train_srcnn import train_srcnn
from Scripts.train_SvOcSrcnn import train_SvOcSRCNN
from Scripts.visualize_results import visualize_results_from_json
from Models.SvOcSRCNN_model import SvOcSRCNN
from Models.SRCNN_model import SRCNN
from Scripts import test_model


def convert_to_builtin(obj):
    """Recursively convert NumPy data types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    download_div2k_flag = False

    # Step 1: Download and prepare dataset
    if download_div2k_flag:
        print("Preparing DIV2K dataset...")
        download_div2k("Data")
    full_dataset = DIV2KDataset("Data/DIV2K", scale=2)

    # Split dataset into train/val/test
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Step 2: Train baseline SRCNN
    srcnn = SRCNN().to(device)
    train_srcnn(
        model=srcnn,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        device=device
    )

    # Step 3: Train SvOcSRCNN with perceptual loss
    print("\nTraining SvOcSRCNN (perceptual loss)...")
    svoc = SvOcSRCNN()
    perceptual_loss_fn = PerceptualLoss(resize=True)
    train_SvOcSRCNN(
        model=svoc,
        train_loader=train_loader,
        val_loader=val_loader,
        perceptual_loss_fn=perceptual_loss_fn,
        num_epochs=15,
        alpha=0.8,
        lr=1e-4,
        device=device,
        save_path="svocsrcnn.pth"
    )

    # Step 4: Test models
    results["SRCNN"] = test_model.evaluate_and_collect(srcnn, "SRCNN", device, test_loader)
    results["SvOcSRCNN"] = test_model.evaluate_and_collect(svoc, "SvOcSRCNN", device, test_loader)

    # Step 5: Save results to JSON
    clean_results = convert_to_builtin(results)
    with open("test_results.json", "w") as f:
        json.dump(clean_results, f, indent=4)

    print("Done. Results saved to test_results.json")

    # Step 6: Visualize results
    print("\nVisualizing results...")
    visualize_results_from_json(clean_results)


if __name__ == "__main__":
    main()
