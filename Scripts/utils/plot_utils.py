from PIL import Image, ImageDraw, ImageFont       # For drawing text and making collages
import torchvision.transforms.functional as TF   # For tensor ↔ image conversion

def annotate_image(tensor_img, text):
    img = TF.to_pil_image(tensor_img.squeeze(0).cpu())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((5, 5), text, fill="white", font=font)
    return TF.to_tensor(img)

def create_collage(images, save_path):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    collage.save(save_path)