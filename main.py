import torch
import wandb
import yaml
from Scripts.pipelines.preprocess_pipeline import preprocess_pipeline
from Scripts.pipelines.train_pipeline import train_pipeline
from Scripts.pipelines.test_pipeline import test_pipeline

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # === Load Config ===
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # === Optional wandb init ===
    if config.get("use_wandb", False):
        wandb.init(project="super-resolution", config=config, name=config["model"])

    # === Step 1: Preprocessing ===
    train_loader, val_loader, test_loader, forced_indices = preprocess_pipeline(config)

    # === Step 2: Training ===
    if config["train"] == True:
        trained_model, history = train_pipeline(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )

    # === Step 3: Testing ===
    if config["test"] == True:
        metrics = test_pipeline(
            config=config,
            test_loader=test_loader,
            forced_indices=forced_indices,
            device=device,
            history=history
        )

    print("✅ All stages complete.")

if __name__ == "__main__":
    main()