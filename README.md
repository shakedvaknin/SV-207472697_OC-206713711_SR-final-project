# Super-Resolution on DIV2K: SRCNN, VDSR, and VDSR_SA
This notebook 
This project performs image super-resolution on the DIV2K dataset using three models:
- **SRCNN** (baseline)
- **VDSR** (baseline)
- **VDSR_SA** (our custom model with Spatial Attention)

The notebook trains and evaluates all models and compares them using PSNR, SSIM, and FID metrics.

## ⚠️ Colab Runtime Notice

When running the notebook on **Google Colab**, some packages (like `piq`, `lpips`, etc.) trigger an **automatic runtime restart** after installation.

> ✅ After the runtime restarts, **you must re-run the notebook from the beginning**.

## 🗂️ Notebook Structure

- **Section 1**: Setup and installation
- **Section 2**: Dataset download and preprocessing
- **Section 3**: Model definitions
- **Section 4**: Training and validation
- **Section 5**: Evaluation metrics and visualizations
- **Section 6**: Comparison and conclusions

© 2025 Shaked Vaknin and Ofek Cohen.
