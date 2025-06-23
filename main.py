import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import random
from collections import OrderedDict


random.seed(42)  # For reproducibility

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data.data_loader import DIV2KDataset
from Data.data_utils import download_div2k
from Data.data_utils import preprocess_div2k
from Data.data_utils import preprocess_div2k_center_crop

from Models.SRCNN_model import SRCNN
from Models.SvOcSRCNN_model import SvOcSRCNN
from Models.VDSR_Attention import VDSR_SA
from Models.VDSR import VDSR
from Models.RCAN import RCAN
from Models.FusionNet import FusionNet

from Scripts.train.train_fusion import train_fusion_net
from Scripts.train.train_val_test import train_val_test
from Scripts.train.train_no_upsample import train_no_upsample
from Scripts.losses import CombinedLoss
from Scripts.losses import CharbonnierLoss
from Scripts.utils.plot_utils import plot_training_curves, generate_summary_collage_from_checkpoints
from Scripts.attention_based_fusion import run_attention_fusion_inference

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1  # Set to 1 for quick testing, increase for actual training
    # Step 1: Download and prepare dataset
    print("Preparing DIV2K dataset...")
    download_div2k("Data")
    # preprocess_div2k(source_folder="Data/DIV2K", target_folder="Data/DIV2K_NORMALIZED")
    # preprocess_div2k_center_crop(source_folder="Data/DIV2K", target_folder="Data/DIV2K_NORMALIZED")
    full_dataset = DIV2KDataset("Data/DIV2K_NORMALIZED", scale=4)

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

    forced_indices = sorted(random.sample(range(80), 10))

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
        num_epochs=1,
        lr=1e-4,
        forced_indices = forced_indices

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
        num_epochs=num_epochs,
        forced_indices = forced_indices
    )
    print("Training complete. SvOcSRCNN model saved.")
    print("Test metrics:", (SvOcSRCNN_test_metrics))

    # Step 4: Train VDSR_SA model
    print("Training VDSR_SA model...")
    vdsr_model = VDSR_SA().to(device)
    loss_fn = nn.MSELoss()
    tained_vdsr_sa_model, vdsr_sa_history, vdsr_sa_test_metrics = train_val_test(
        model=vdsr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_dir="checkpoints/vdsr_sa",
        num_epochs=num_epochs,
        lr=1e-4,
        forced_indices = forced_indices
    )
    print("Training complete. VDSR_SA model saved.")
    print("Test metrics:", (vdsr_sa_test_metrics))

    
    # === Train VDSR ===
    vdsr = VDSR(num_channels=3)
    vdsr_loss = CharbonnierLoss()
    print("\nStarting VDSR Training...")
    trained_vdsr_model, vdsr_history, vdsr_test_metrics = train_val_test(
        model=vdsr,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=vdsr_loss,
        lr=1e-4,
        num_epochs=num_epochs,
        device=device,
        save_dir="checkpoints/vdsr",
        verbose=True,
        forced_indices=forced_indices
    )

    print("Training complete. VDSR model saved.")
    print("Test metrics:", (vdsr_test_metrics))

    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()

    # === RCAN Setup ===
    rcan_model = RCAN(num_channels=3)
    optimizer = torch.optim.Adam(rcan_model.parameters(), lr=1e-4)
    loss_fn = CombinedLoss(alpha=0.95, device=device)

    # === Training ===
    print("Starting RCAN Training...")
    trained_rcan_model, rcan_history, rcan_test_metrics = train_no_upsample(
        model=rcan_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        model_name="RCAN",
        save_dir="checkpoints/rcan",
        num_epochs=num_epochs,
        val_fid_interval=5,
        forced_indices=None,
        device=device,
        verbose=True
    )
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained models
    trained_vdsr_model = VDSR(num_channels=3)
    trained_vdsr_model.load_state_dict(torch.load("checkpoints/VDSR/best_model.pth", map_location=device))

    # Create model instance with correct args
    trained_rcan_model = RCAN(num_channels=3, n_resgroups=5, n_RCAB=10)  # Adjust args to match your definition exactly

    # Load and clean checkpoint
    ckpt_path = "checkpoints/RCAN/best_model.pth"
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # If the state_dict contains 'model_state_dict', adjust
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Remove 'module.' prefix if present
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        clean_state_dict[new_k] = v

    trained_rcan_model.load_state_dict(clean_state_dict)

    # Initialize FusionNet and training components
    fusion_net = FusionNet(in_channels=6)
    optimizer = torch.optim.Adam(fusion_net.parameters(), lr=1e-4)
    loss_fn = CombinedLoss(alpha=0.8, device=device)

    # Use your own DataLoader here

    print("Starting FusionNet Training...")
    train_fusion_net(
        fusion_net=fusion_net,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        background_model=trained_vdsr_model,
        object_model=trained_rcan_model,
        fusion_ckpt_dir="checkpoints/FusionNet",
        scale=2,
        num_channels=3,
        num_epochs=num_epochs,
        device=device
    )
    print("FusionNet training complete.")
    print("\nAll models trained. Running attention-based fusion inference...")

    plot_training_curves(srcnn_history, "checkpoints/srcnn_mse/training_plot.png")
    plot_training_curves(SvOcSRCNN_history, "checkpoints/svoc-perceptual/training_plot.png")
    plot_training_curves(vdsr_sa_history, "checkpoints/vdsr_sa/training_plot.png")
    plot_training_curves(vdsr_history, "checkpoints/vdsr/training_plot.png")
    plot_training_curves(rcan_history, "checkpoints/rcan/training_plot.png")
    # === Run Fusion Inference ===
    run_attention_fusion_inference(test_loader=test_loader, device=device)

    generate_summary_collage_from_checkpoints()

    
if __name__ == "__main__":
    main()
