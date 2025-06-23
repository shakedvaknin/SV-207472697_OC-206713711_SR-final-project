# main.py
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import random
import wandb
import yaml
from pathlib import Path

from Data.data_loader import DIV2KDataset
from Data.data_utils import download_div2k

from Models.SRCNN_model import SRCNN
from Models.SvOcSRCNN_model import SvOcSRCNN
from Models.VDSR_Attention import VDSR_SA
from Models.VDSR import VDSR
from Models.RCAN import RCAN
from Models.FusionNet import FusionNet

from Scripts.train.train_val_test import train_val_test
from Scripts.train.train_no_upsample import train_no_upsample
from Scripts.train.train_fusion import train_fusion_net
from Scripts.losses import CombinedLoss, CharbonnierLoss
from Scripts.utils.plot_utils import plot_training_curves, generate_summary_collage_from_checkpoints
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
    else:
        raise ValueError("Unsupported loss")

def main():
    args = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args["use_wandb"]:
        wandb.init(project="super-resolution", config=args, name=args["model"])

    print("Preparing DIV2K dataset...")
    download_div2k("Data")
    full_dataset = DIV2KDataset("Data/DIV2K_NORMALIZED", scale=args["scale"])

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    forced_indices = sorted(random.sample(range(len(test_dataset)), 10))
    loss_fn = get_loss_fn(args["loss"], device)

    if args["model"] == "SRCNN":
        model = SRCNN().to(device)
        trainer = train_val_test
    elif args["model"] == "SvOcSRCNN":
        model = SvOcSRCNN().to(device)
        trainer = train_val_test
    elif args["model"] == "VDSR_SA":
        model = VDSR_SA().to(device)
        trainer = train_val_test
    elif args["model"] == "VDSR":
        model = VDSR(num_channels=3).to(device)
        trainer = train_val_test
    elif args["model"] == "RCAN":
        model = RCAN(num_channels=3, scale=args["scale"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trained_model, history, metrics = train_no_upsample(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            model_name="RCAN",
            save_dir=args["save_dir"],
            num_epochs=args["epochs"],
            device=device,
            forced_indices=forced_indices,
            verbose=True
        )
        if args["use_wandb"]:
            wandb.log(metrics)
        log_result(args["model"], args["loss"], metrics, args["save_dir"])
        return
    elif args["model"] == "FusionNet":
        print("FusionNet should be trained after VDSR and RCAN.")
        return

    trained_model, history, metrics = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_dir=args["save_dir"],
        num_epochs=args["epochs"],
        device=device,
        forced_indices=forced_indices
    )

    if args["use_wandb"]:
        wandb.log(metrics)
    log_result(args["model"], args["loss"], metrics, args["save_dir"])
    # plot_training_curves(history, Path(args["save_dir"]) / "training_plot.png")
    generate_summary_collage_from_checkpoints()

if __name__ == "__main__":
    main()
