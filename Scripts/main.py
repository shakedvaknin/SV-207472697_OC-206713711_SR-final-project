from imports import *
from helpers import download_div2k, PerceptualLoss
from Data.data_loader import DIV2KDataset
from Scripts import train_srcnn, train_SvOcSRCNN, visualize_results_from_json #visualize_results
from Models import SvOcSRCNN, SRCNN
from Scripts import test_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # Step 1: Download and prepare dataset
    print("Preparing DIV2K dataset...")
    download_div2k("data")
    full_dataset = DIV2KDataset("data/DIV2K_train_HR", scale=2, max_images=60)

    # Split dataset into train/val
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # Ensures total = 100%

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


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

    # Step 5 - Test the models
    results["SRCNN"] = test_model.evaluate_and_collect(SRCNN.srcnn, "SRCNN", device,test_loader)
    results["SvOcSRCNN"] = test_model.evaluate_and_collect(SvOcSRCNN.svoc, "SvOcSRCNN", device,test_loader)

    # Save to JSON
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Done. Results saved to test_results.json")

    # Step 5: Visualize comparison
    print("\nVisualizing results...")

    with open("test_results.json", "r") as f:
        results = json.load(f)

    visualize_results_from_json(results)


if __name__ == "__main__":
    main()