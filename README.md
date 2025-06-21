# Super-Resolution CNN Final Project

This project explores image super-resolution using deep learning techniques. The core focus is on enhancing low-resolution images (4x downsampled) using several CNN-based architectures trained and evaluated on the DIV2K dataset.

## 📌 Project Structure

```
├── main.py                      # Main training script
├── Data/                       # Folder for storing DIV2K data
├── Models/                    # Implementations of SRCNN, SvOcSRCNN, VDSR-SA
├── Scripts/
│   ├── train_val_test.py       # Training, validation, testing loop
│   ├── losses.py               # Perceptual and combined loss functions
│   └── utils/                  # Utility functions (metrics, visualization, etc.)
└── checkpoints/                # Saved model weights and results
```

## 🧠 Implemented Models

- **SRCNN**: Basic 3-layer convolutional neural network.
- **SvOcSRCNN**: Enhanced SRCNN variant with multiple skip connections.
- **VDSR-SA**: Very Deep Super-Resolution model with spatial attention modules.

## 📊 Loss Functions

- **MSELoss**: Default pixel-wise loss.
- **Perceptual Loss**: Based on VGG19 features.
- **Combined Loss**: Weighted sum of perceptual and MSE losses.

## 📈 Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **FID**: Frechet Inception Distance (for visual quality comparison)

## 🖼️ Visualization

Collage-style output is generated for selected test samples showing:
- Low-resolution input (LR)
- Super-resolved output (SR)
- Ground truth high-resolution image (GT)

Each panel is annotated with the corresponding PSNR and SSIM values.

## 🛠️ How to Run

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run training:

    ```bash
    python main.py
    ```

3. Models and evaluation outputs will be saved in `checkpoints/<model_name>/`.

## 📦 Requirements

See `requirements.txt` for detailed package versions.

## 🧪 Notes

- FID is calculated using `torchmetrics`' Frechet Inception Distance.
- Collages are saved instead of individual images to save disk space.
- `VDSR-SA` uses spatial attention to improve local detail restoration.

---

### ✅ Contributions

Built with ❤️ by [Ofek] as part of a deep learning project on image super-resolution.