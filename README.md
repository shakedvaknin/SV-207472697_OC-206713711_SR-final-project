# Hyper-Resolution Image Super-Resolution Project

This project provides a unified framework for training, evaluating, and comparing deep learning models for single image super-resolution (SISR) using the DIV2K dataset. It supports classical architectures like SRCNN, VDSR, and RCAN, as well as custom variants like SvOcSRCNN and FusionNet.

---

## 📁 Project Structure

```
.
├── main.py                    # Entry point for training and evaluation
├── config/config.yaml         # Configuration file (paths, hyperparameters)
├── checkpoints/               # Saved model weights
├── Data/
│   ├── data_loader.py         # Custom PyTorch Dataset & DataLoader
│   └── data_utils.py          # Image processing utilities
├── Models/                    # Model architectures (SRCNN, VDSR, RCAN, etc.)
├── Scripts/
│   ├── train.py               # Unified training/validation/testing loop
│   ├── train_srcnn.py        # Model-specific training (e.g., SRCNN)
│   ├── train_rcan.py         # RCAN-style model training
│   ├── visualize_results.py  # Result plotting & collage generation
│   └── utils/                 # Metric utilities (PSNR, SSIM), plotting, etc.
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hyper-resolution.git
cd hyper-resolution
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Ensure you have an NVIDIA GPU and CUDA toolkit installed for best performance.

---

## 📦 Dataset

This project uses the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

Download the high-resolution images and place them in:

```
Data/DIV2K/
```

A preprocessing script will automatically resize and normalize images.

---

## ⚙️ Configuration

Edit `config/config.yaml` to set:
- Paths to training/validation/test data
- Model name
- Training parameters (epochs, batch size, loss)
- Device settings
- Optional logging

---

## 🏋️‍♂️ Training & Evaluation

### Train a model
```bash
python main.py --model SRCNN
```

### Train RCAN-style model
```bash
python main.py --model RCAN --no_upsample
```

### Supported models
- `SRCNN`
- `SvOcSRCNN`
- `VDSR`
- `VDSR_SA`
- `RCAN`
- `FusionNet`

---

## 📊 Metrics

Evaluation uses:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **FID (Fréchet Inception Distance)** *(if enabled)*

Results are logged to CSV and visualized using:
- Per-image annotated outputs
- Collages comparing multiple models

---

## 🖼 Example Outputs

<img src="results/collage_example.png" width="600">

---

## 🛠 Troubleshooting

- Ensure `torch` is GPU-enabled
- Clamp outputs with `.clamp(0, 1)` to avoid visual artifacts
- Validate model paths and image dimensions

---

## 📄 License

This project is for academic/research use only. For commercial use, please contact the author.

---