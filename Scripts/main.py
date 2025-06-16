from imports import *
from helpers import download_div2k, PerceptualLoss
from Data.data_loader import DIV2KDataset
from Scripts import train_srcnn, train_SvOcSRCNN, visualize_results
from Models import SvOcSRCNN, SRCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Download and prepare dataset
    print("Preparing DIV2K dataset...")
    download_div2k("data")
    full_dataset = DIV2KDataset("data/DIV2K_train_HR", scale=2, max_images=60)

    # Split dataset into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Step 2: Train SRCNN
    print("\nTraining SRCNN...")
    srcnn_model, srcnn_history = train_srcnn(
        SRCNN, DIV2KDataset, download_div2k,
        scale=2, max_images=60,
        batch_size=4, num_epochs=15, device=device
    )

    # Step 3: Train SvOcSRCNN with perceptual loss
    print("\nTraining SvOcSRCNN (perceptual loss)...")
    svoc_model = SvOcSRCNN()
    perceptual_loss_fn = PerceptualLoss(resize=True)
    svoc_history = train_SvOcSRCNN(
        model=svoc_model,
        train_loader=train_loader,
        val_loader=val_loader,
        perceptual_loss_fn=perceptual_loss_fn,
        num_epochs=15,
        alpha=0.8,
        lr=1e-4,
        device=device,
        save_path="svocsrcnn.pth"
    )

    # Step 4: Visualize comparison
    print("\nVisualizing results...")
    visualize_results(
        models=[srcnn_model, svoc_model],
        #models=[svoc_model],
        model_names=["SRCNN", "SvOcSRCNN"],
        #model_names=["SvOcSRCNN"],
        dataset=full_dataset,
        device=device,
        num_images=5
    )

if __name__ == "__main__":
    main()