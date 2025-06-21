import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data.data_loader import DIV2KDataset
from Data.data_utils import download_div2k

from Models.SRCNN_model import SRCNN
from Models.SvOcSRCNN_model import SvOcSRCNN
from Models.VDSR_Attention import VDSR_SA
from Models.VDSR import VDSR
from Models.RCAN import RCAN

from Scripts.train_val_test import train_val_test
from Scripts.losses import CombinedLoss
from Scripts.losses import CharbonnierLoss
from Scripts.utils.plot_utils import plot_training_curves

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Step 2: Train SRCNN model
    print("Training SRCNN model...")
    srcnn = SRCNN().to(device)
    # Use plain MSE loss
    loss_fn = nn.MSELoss()

    trained_srcnn_model, srcnn_history, srcnn_test_metrics = train_val_test(
        model=srcnn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_dir="checkpoints/srcnn_mse",
        num_epochs=15,
        lr=1e-4
    )
    print("Training complete. SRCNN model saved.")
    print("Test metrics:", (srcnn_test_metrics))

    # Step 3: Train SvOcSRCNN model
    model = SvOcSRCNN()
    loss_fn = CombinedLoss(alpha=0.8, device=device)
    print("Training SvOcSRCNN model...")
    trained_SvOcSRCNN_model, SvOcSRCNN_history, SvOcSRCNN_test_metrics = train_val_test(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_dir="checkpoints/svoc-perceptual",
        num_epochs=15
    )
    print("Training complete. SvOcSRCNN model saved.")
    print("Test metrics:", (SvOcSRCNN_test_metrics))

    # Step 4: Train VDSR_SA model
    print("Training VDSR_SA model...")
    vdsr_model = VDSR_SA().to(device)
    loss_fn = nn.MSELoss()
    tained_vdsr_model, vdsr_history, vdsr_test_metrics = train_val_test(
        model=vdsr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_dir="checkpoints/vdsr_sa",
        num_epochs=15,
        lr=1e-4
    )
    print("Training complete. VDSR_SA model saved.")
    print("Test metrics:", (vdsr_test_metrics))


    # === Train VDSR ===
    vdsr = VDSR(num_channels=3)
    vdsr_loss = CharbonnierLoss()
    print("\nStarting VDSR Training...")
    train_val_test(
        model=vdsr,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=vdsr_loss,
        lr=1e-4,
        num_epochs=15,
        device=device,
        save_dir="checkpoints/VDSR",
        verbose=True
    )

    # === Train RCAN ===
    rcan = RCAN(scale=2, num_channels=3)
    rcan_loss = CombinedLoss(alpha=0.8, device=device)
    print("\nStarting RCAN Training...")
    train_val_test(
        model=rcan,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=rcan_loss,
        lr=1e-4,
        num_epochs=15,
        device=device,
        save_dir="checkpoints/RCAN",
        verbose=True
    )  

    print("\nBoth models trained. Ready for fusion inference.")


    plot_training_curves(srcnn_history, "checkpoints/srcnn_mse/training_plot.png")
    plot_training_curves(SvOcSRCNN_history, "checkpoints/svoc-perceptual/training_plot.png")
    plot_training_curves(vdsr_history, "checkpoints/vdsr_sa/training_plot.png")
    
if __name__ == "__main__":
    main()
