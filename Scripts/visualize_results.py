from imports import *
import matplotlib.pyplot as plt


def visualize_results_from_json(results_json):

    model_names = list(results_json.keys())

    psnr_scores = [results_json[m]["PSNR"] for m in model_names]
    ssim_scores = [results_json[m]["SSIM"] for m in model_names]
    fid_scores  = [results_json[m]["FID"] for m in model_names]

    x = range(len(model_names))

    plt.figure(figsize=(15, 4))

    # PSNR Bar Chart
    plt.subplot(1, 3, 1)
    plt.bar(x, psnr_scores, color='skyblue')
    plt.xticks(x, model_names)
    plt.ylabel("PSNR")
    plt.title("PSNR (↑ better)")
    plt.grid(axis='y')

    # SSIM Bar Chart
    plt.subplot(1, 3, 2)
    plt.bar(x, ssim_scores, color='lightgreen')
    plt.xticks(x, model_names)
    plt.ylabel("SSIM")
    plt.title("SSIM (↑ better)")
    plt.grid(axis='y')

    # FID Bar Chart
    plt.subplot(1, 3, 3)
    plt.bar(x, fid_scores, color='salmon')
    plt.xticks(x, model_names)
    plt.ylabel("FID")
    plt.title("FID (↓ better)")
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()
