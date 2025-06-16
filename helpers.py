from imports import *

def download_div2k(destination="data"):
    os.makedirs(destination, exist_ok=True)
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    filename = os.path.join(destination, "DIV2K_train_HR.zip")

    if not os.path.exists(filename):
        print("Downloading DIV2K...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
    else:
        print("DIV2K already downloaded.")

    # Extract
    extract_path = os.path.join(destination, "DIV2K_train_HR")
    if not os.path.exists(extract_path):
        print("Extracting DIV2K...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print("Done.")
    else:
        print("DIV2K already extracted.")

def save_batch_as_images(batch_tensor, root_dir):
    """Save a batch of images to the specified directory."""
    for i, img in enumerate(batch_tensor):
        img = img.clamp(0, 1).cpu()
        vutils.save_image(img, os.path.join(root_dir, f"{uuid.uuid4().hex}.png"))

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:16].eval()  # up to conv3_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # VGG normalization
            std=[0.229, 0.224, 0.225]
        )
        self.resize = resize

    def forward(self, sr, hr):
        # VGG expects [0,1] images normalized to ImageNet stats
        sr = self.transform(sr.clamp(0, 1))
        hr = self.transform(hr.clamp(0, 1))

        if self.resize:
            sr = F.interpolate(sr, size=(224, 224), mode='bilinear', align_corners=False)
            hr = F.interpolate(hr, size=(224, 224), mode='bilinear', align_corners=False)

        f1 = self.vgg(sr)
        f2 = self.vgg(hr)

        return F.l1_loss(f1, f2)

def total_loss(sr, hr, perceptual_loss_fn, alpha=0.8):
    l1 = F.l1_loss(sr, hr)
    perceptual = perceptual_loss_fn(sr, hr)
    return alpha * l1 + (1 - alpha) * perceptual