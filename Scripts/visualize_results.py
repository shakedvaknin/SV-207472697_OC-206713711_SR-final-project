from imports import *

def visualize_results(models, model_names, dataset, device, num_images=5):
    assert len(models) == len(model_names), "Each model must have a corresponding name."

    for model in models:
        model.eval()

    with torch.no_grad():
        for i in range(num_images):
            lr, hr = dataset[i]
            lr_input = lr.unsqueeze(0).to(device)

            outputs = []
            for model in models:
                sr = model(lr_input).squeeze(0).cpu()
                outputs.append(sr)

            # Create subplot: LR + N models + HR
            total_cols = len(models) + 2
            fig, axs = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))

            axs[0].imshow(np.transpose(lr.numpy(), (1, 2, 0)))
            axs[0].set_title("Low-Res Input")
            axs[0].axis('off')

            for j, sr in enumerate(outputs):
                axs[j + 1].imshow(np.transpose(sr.numpy(), (1, 2, 0)))
                axs[j + 1].set_title(model_names[j])
                axs[j + 1].axis('off')

            axs[-1].imshow(np.transpose(hr.numpy(), (1, 2, 0)))
            axs[-1].set_title("Ground Truth (HR)")
            axs[-1].axis('off')

            plt.tight_layout()
            plt.show()