import csv
import os
import time
from datetime import datetime

def log_result(
    model_name,
    loss_type,
    metrics,
    save_dir,
    final_train_loss=None,
    final_val_loss=None,
    csv_path="results.csv"
):
    """Append model results to a central CSV log with timestamps and losses."""
    fieldnames = [
        "timestamp_unix",
        "timestamp_readable",
        "model",
        "loss",
        "train_loss",
        "val_loss",
        "psnr",
        "ssim",
        "fid",
        "save_dir"
    ]
    file_exists = os.path.exists(csv_path)

    timestamp_unix = int(time.time())
    timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp_unix": timestamp_unix,
        "timestamp_readable": timestamp_readable,
        "model": model_name,
        "loss": loss_type,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "psnr": metrics.get("test_psnr"),
        "ssim": metrics.get("test_ssim"),
        "fid": metrics.get("test_fid"),
        "save_dir": save_dir
    }

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
