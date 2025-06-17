from imports import *

# def visualize_results(models, model_names, dataset, device, num_images=5):
#     assert len(models) == len(model_names), "Each model must have a corresponding name."

#     for model in models:
#         model.eval()

#     with torch.no_grad():
#         for i in range(num_images):
#             lr, hr = dataset[i]
#             lr_input = lr.unsqueeze(0).to(device)

#             outputs = []
#             for model in models:
#                 sr = model(lr_input).squeeze(0).cpu()
#                 outputs.append(sr)

#             # Create subplot: LR + N models + HR
#             total_cols = len(models) + 2
#             fig, axs = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))

#             axs[0].imshow(np.transpose(lr.numpy(), (1, 2, 0)))
#             axs[0].set_title("Low-Res Input")
#             axs[0].axis('off')

#             for j, sr in enumerate(outputs):
#                 axs[j + 1].imshow(np.transpose(sr.numpy(), (1, 2, 0)))
#                 axs[j + 1].set_title(model_names[j])
#                 axs[j + 1].axis('off')

#             axs[-1].imshow(np.transpose(hr.numpy(), (1, 2, 0)))
#             axs[-1].set_title("Ground Truth (HR)")
#             axs[-1].axis('off')

#             plt.tight_layout()
#             plt.show()


def visualize_results_from_json(results_json, num_images=5):
    model_names = list(results_json.keys())
    num_models = len(model_names)

    # --- Plot example images ---
    for i in range(num_images):
        total_cols = num_models + 2  # LR + HR + each model
        fig, axs = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))

        # Use LR and HR from the first model (assume same input across models)
        lr = results_json[model_names[0]]["images"][i]["lr"]
        hr = results_json[model_names[0]]["images"][i]["hr"]
        axs[0].imshow(np.transpose(lr, (1, 2, 0)))
        axs[0].set_title("Low-Res Input")
        axs[0].axis('off')

        for j, model_name in enumerate(model_names):
            sr = results_json[model_name]["images"][i]["sr"]
            axs[j + 1].imshow(np.transpose(sr, (1, 2, 0)))
            axs[j + 1].set_title(model_name)
            axs[j + 1].axis('off')

        axs[-1].imshow(np.transpose(hr, (1, 2, 0)))
        axs[-1].set_title("Ground Truth (HR)")
        axs[-1].axis('off')
        plt.tight_layout()
        plt.show()

    # --- Plot metrics (PSNR and SSIM) ---
    for metric in ["psnr", "ssim"]:
        plt.figure(figsize=(10, 5))
        for model_name in model_names:
            values = results_json[model_name]["metrics"][metric]
            plt.plot(values, label=model_name)
        plt.title(metric.upper() + " across Test Images")
        plt.xlabel("Image Index")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Plot FID as bar chart ---
    plt.figure(figsize=(6, 4))
    fids = [results_json[m]["metrics"]["fid"] for m in model_names]
    plt.bar(model_names, fids, color=["skyblue", "lightgreen"])
    plt.ylabel("FID Score")
    plt.title("FID Comparison")
    plt.grid(True, axis='y')
    plt.show()