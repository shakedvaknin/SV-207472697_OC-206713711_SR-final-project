from torch import nn
from torch.optim import Adam

from Scripts.train.train_val_test_upsample import train_and_validate
from Scripts.train.train_no_upsample import train_and_validate_no_upsample as train_no_upsample

from Scripts.utils.losses import CombinedLoss, CharbonnierLoss, NewCombinedLoss

from Models.SRCNN import SRCNN
from Models.SvOcSRCNN import SvOcSRCNN
from Models.VDSR import VDSR
from Models.VDSR_Attention import VDSR_SA
from Models.dsrcnn_ca import PyramidDeepSRCNN_CA
from Models.RCAN import RCAN
from Models.RCAN_SWIN import RCAN_Swin


def get_loss_fn(name,device=None):
    if name == "mse":
        return nn.MSELoss()
    elif name == "charbonnier":
        return CharbonnierLoss()
    elif name == "combined":
        return CombinedLoss(alpha=0.8, device=device)
    elif name == "NewCombinedLoss":
        return NewCombinedLoss(alpha=0.2, beta=0.6)
    else:
        raise ValueError("Unsupported loss function: " + name)

def train_pipeline(config, train_loader, val_loader, device):

    model_name = config["model"]
    loss_fn = get_loss_fn(config["loss"], device)
    lr = float(config.get("lr", 1e-4))

    # === Model Selection ===
    if model_name == "SRCNN":
        model = SRCNN().to(device)
    elif model_name == "SvOcSRCNN":
        model = SvOcSRCNN().to(device)
    elif model_name == "VDSR_SA":
        model = VDSR_SA(num_features=64, num_resblocks=24).to(device)
    elif model_name == "VDSR":
        model = VDSR(num_channels=3).to(device)
    elif model_name == "dsrcnn_ca":
        model = PyramidDeepSRCNN_CA(num_channels=3).to(device)
    elif model_name == "RCAN_SWIN":
        model = RCAN_Swin(num_channels=3).to(device)
    elif model_name == "RCAN":
        model = RCAN(num_channels=3, scale=config["scale"]).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported in train_pipeline.")
    
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"Training model: {model_name} with loss: {config['loss']}")

    if model_name in ["RCAN", "RCAN_SWIN"]:
        trained_model, history, _ = train_no_upsample(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,  # not used in training phase
            optimizer=optimizer,
            loss_fn=loss_fn,
            model_name=model_name,
            save_dir=config["save_dir"],
            checkpoint_dir="checkpoints",
            num_epochs=config["epochs"],
            device=device,
            forced_indices=None,
            verbose=True,
            early_stopping_patience=10
        )
    else:
        trained_model, history = train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            save_dir=config["save_dir"],
            checkpoint_dir="checkpoints",
            model_name=model_name,
            num_epochs=config["epochs"],
            device=device,
            use_wandb=config.get("use_wandb", False),
            early_stopping_patience=10
        )

    print(f"Training completed for model: {model_name}")
    return trained_model, history
