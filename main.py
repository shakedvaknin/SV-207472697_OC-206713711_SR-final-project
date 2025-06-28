# main.py
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import random
import wandb
import yaml
from pathlib import Path

from Data.data_loader import DIV2KDataset
from Data.data_utils import download_div2k, preprocess_div2k_center_crop
from Data.data_utils import preprocess_div2k

from Models.SRCNN_model import SRCNN
from Models.SvOcSRCNN_model import SvOcSRCNN
from Models.VDSR_Attention import VDSR_SA
from Models.VDSR import VDSR
from Models.RCAN import RCAN
from Models.FusionNet import FusionNet
from Models.dsrcnn_ca import PyramidDeepSRCNN_CA
from Models.RCAN_SWIN import RCAN_Swin

from Scripts.train.train_val_test import train_val_test
from Scripts.train.train_no_upsample import train_no_upsample
from Scripts.train.train_fusion import train_fusion_net
from Scripts.utils.losses import CombinedLoss, CharbonnierLoss, NewCombinedLoss
from Scripts.utils.plot_utils import generate_summary_collage_from_checkpoints
from Scripts.attention_based_fusion import run_attention_fusion_inference
from Scripts.utils.result_logger import log_result

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_loss_fn(name, device):
    if name == "mse":
        return nn.MSELoss()
    elif name == "charbonnier":
        return CharbonnierLoss()
    elif name == "combined":
        return CombinedLoss(alpha=0.8, device=device)
    elif name == "NewCombinedLoss":
        return NewCombinedLoss(alpha=0.2, beta = 0.6)
    else:
        raise ValueError("Unsupported loss")

def main():
    args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    if args["use_wandb"]:
        wandb.init(project="super-resolution", config=args, name=args["model"])
    # Download DIV2K dataset if not already present
    print("Preparing DIV2K dataset...")
    download_div2k("Data")
    # Preprocess the DIV2K dataset
    full_dataset = DIV2KDataset("Data/DIV2K", scale=args["scale"])
    # Size of each dataset split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    # Splitting the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Indicies for test samples to present
    forced_indices = args["idx"]
    loss_fn = get_loss_fn(args["loss"], device)
    # Initialize the model based on the configuration
    if args["model"] == "SRCNN":
        model = SRCNN().to(device)
        trainer = train_val_test
    elif args["model"] == "SvOcSRCNN":
        model = SvOcSRCNN().to(device)
        trainer = train_val_test
    elif args["model"] == "VDSR_SA":
        model = VDSR_SA(num_features=64, num_resblocks=24).to(device)
        trainer = train_val_test
    elif args["model"] == "VDSR":
        model = VDSR(num_channels=3).to(device)
        trainer = train_val_test
    elif args["model"] == "dsrcnn_ca":
        model = PyramidDeepSRCNN_CA(num_channels=3).to(device)
        trainer = train_val_test
    elif args["model"] == "RCAN_SWIN":
        model = RCAN_Swin(num_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        trainer = train_no_upsample
        print("Training RCAN_SWIN without upsampling...")
        trained_model, history, metrics = train_no_upsample(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            model_name=args["model"],
            save_dir=args["save_dir"],
            num_epochs=args["epochs"],
            device=device,
            forced_indices=forced_indices,
            verbose=True
        )
        print("RCAN_SWIN training completed.")
        if args["use_wandb"]:
            wandb.log(metrics)
        log_result(args["model"], args["loss"], metrics, args["save_dir"])
        generate_summary_collage_from_checkpoints()
        return
    elif args["model"] == "RCAN":
        model = RCAN(num_channels=3, scale=args["scale"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # If the model is RCAN, we need to train it without upsampling
        print("Training RCAN without upsampling...")
        trained_model, history, metrics = train_no_upsample(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            model_name=args["model"],
            save_dir=args["save_dir"],
            num_epochs=args["epochs"],
            device=device,
            forced_indices=forced_indices,
            verbose=True
        )
        print("RCAN training completed.")
        if args["use_wandb"]:
            wandb.log(metrics)
        log_result(args["model"], args["loss"], metrics, args["save_dir"])
        generate_summary_collage_from_checkpoints()
        return
    # If the model is FusionNet, we need to load VDSR and RCAN checkpoints
    elif args["model"] == "FusionNet":
        vdsr_ckpt = Path("checkpoints/vdsr_best_model.pth")
        rcan_ckpt = Path("checkpoints/RCAN_best_model.pth")
        if not vdsr_ckpt.exists() or not rcan_ckpt.exists():
            print("❌ Required checkpoints for VDSR and/or RCAN not found. Train them first.")
            return
        print("✅ Found checkpoints. Loading VDSR and RCAN...")
        vdsr_model = VDSR(num_channels=3)
        vdsr_model.load_state_dict(torch.load(vdsr_ckpt, map_location=device))
        rcan_model = RCAN(num_channels=3, scale=args["scale"])
        rcan_model.load_state_dict(torch.load(rcan_ckpt, map_location=device))
        fusion_model = FusionNet(in_channels=6, out_channels=3)  # 3 from VDSR + 3 from RCAN
        fusion_model.to(device)
        fusion_optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-4)
        fusion_loss_fn = get_loss_fn(args["loss"], device)
        print("✅ Loaded VDSR and RCAN models.")
        train_fusion_net(
            fusion_net=fusion_model,
            train_loader=train_loader,
            optimizer=fusion_optimizer,
            loss_fn=fusion_loss_fn,
            background_model=rcan_model,
            object_model=vdsr_model,
            fusion_ckpt_dir=args["save_dir"],
            scale=args["scale"],
            device=device,
            num_epochs=args["epochs"]
        )
        print("FusionNet training completed.")
        generate_summary_collage_from_checkpoints()
        return

    # If the model is not FusionNet or RCAN, we proceed with the standard training
    print(f"Training model: {args['model']} with loss: {args['loss']}")

    trained_model, history, metrics = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=args["model"],
        loss_fn=loss_fn,
        save_dir=args["save_dir"],
        num_epochs=args["epochs"],
        device=device,
        forced_indices=forced_indices
    )
    print(f"Training completed for model: {args['model']}")

    if args["use_wandb"]:
        wandb.log(metrics)
    log_result(args["model"], args["loss"], metrics, args["save_dir"])
    
    generate_summary_collage_from_checkpoints()

if __name__ == "__main__":
    main()
