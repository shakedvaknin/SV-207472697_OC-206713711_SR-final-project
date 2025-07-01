from Scripts.train.train_val_test_upsample import test_upsample as test_model_with_upsample
from Scripts.train.train_no_upsample import test_no_upsample as test_model_no_upsample
from Scripts.utils.result_logger import log_result
from Scripts.utils.plot_utils import generate_summary_collage_from_checkpoints
from Scripts.utils.result_logger import log_result

from Models.SRCNN import SRCNN
from Models.SvOcSRCNN import SvOcSRCNN
from Models.VDSR import VDSR
from Models.VDSR_Attention import VDSR_SA
from Models.dsrcnn_ca import PyramidDeepSRCNN_CA
from Models.RCAN import RCAN
from Models.RCAN_SWIN import RCAN_Swin

import wandb

def test_pipeline(config, test_loader, forced_indices, device, history=None):
    model_name = config["model"]

    # === Model Instantiation ===
    if model_name == "SRCNN":
        model = SRCNN()
        test_fn = test_model_with_upsample
    elif model_name == "SvOcSRCNN":
        model = SvOcSRCNN()
        test_fn = test_model_with_upsample
    elif model_name == "VDSR":
        model = VDSR(num_channels=3)
        test_fn = test_model_with_upsample
    elif model_name == "VDSR_SA":
        model = VDSR_SA(num_features=64, num_resblocks=24)
        test_fn = test_model_with_upsample
    elif model_name == "dsrcnn_ca":
        model = PyramidDeepSRCNN_CA(num_channels=3)
        test_fn = test_model_with_upsample
    elif model_name == "RCAN":
        model = RCAN(num_channels=3, scale=config["scale"])
        test_fn = test_model_no_upsample
    elif model_name == "RCAN_SWIN":
        model = RCAN_Swin(num_channels=3)
        test_fn = test_model_no_upsample
    else:
        raise ValueError(f"Unsupported model in test_pipeline: {model_name}")

    model = model.to(device)

    # === Run the test function ===
    metrics, example_data = test_fn(
        model=model,
        test_loader=test_loader,
        save_dir=config["save_dir"],
        checkpoint_dir="checkpoints",
        model_name=model_name,
        forced_indices=forced_indices,
        device=device,
        use_wandb=config.get("use_wandb", False),
        verbose=True
    )

    if config.get("use_wandb", False):
        wandb.log(metrics)

    log_result(model_name, config["loss"], metrics, config["save_dir"])
    generate_summary_collage_from_checkpoints()

    print(f"Testing complete for model: {model_name}")

    final_train_loss = history['train_loss'][-1] if history and 'train_loss' in history else None
    final_val_loss = history['val_loss'][-1] if history and 'val_loss' in history else None

    log_result(
    model_name=config["model"],
    loss_type=config["loss"],
    metrics=metrics,
    save_dir=config["save_dir"],
    final_train_loss=final_train_loss,
    final_val_loss=final_val_loss
    )
    
    return metrics
