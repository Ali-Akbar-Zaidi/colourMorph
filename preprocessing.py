import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

# Must match style_transfer.py. VGG19 was trained with this normalization.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BILINEAR = Image.Resampling.BILINEAR
ROOT = Path(__file__).resolve().parent


def preprocess_image(image, image_size=256, device=None):
    """Return a normalized RGB tensor of shape [1, 3, image_size, image_size]."""
    image = image.convert("RGB")
    image = image.resize((image_size, image_size), BILINEAR)
    tensor = TF.to_tensor(image)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def denormalize_image(image_tensor):
    """Reverse ImageNet normalization and return a PIL RGB image."""
    tensor = image_tensor.detach().cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor * std + mean).clamp(0, 1)
    return TF.to_pil_image(tensor)


def _sample_photo():
    folders = [
        ROOT / "sample_images" / "content",
        ROOT / "sample_images" / "style",
        ROOT / "sample_images",
    ]
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
    return None


def _color_blocks(size=320):
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    half = size // 2
    draw.rectangle((0, 0, half, half), fill=(220, 40, 40))
    draw.rectangle((half, 0, size, half), fill=(40, 180, 60))
    draw.rectangle((0, half, half, size), fill=(40, 80, 220))
    draw.rectangle((half, half, size, size), fill=(240, 200, 40))
    return image


def main():
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        image = Image.open(source)
        print(f"source {source}")
    else:
        sample = _sample_photo()
        if sample is not None:
            image = Image.open(sample)
            print(f"source {sample}")
        else:
            image = _color_blocks()
            print("source generated colour blocks")

    resized = image.convert("RGB").resize((256, 256), BILINEAR)
    tensor = preprocess_image(image, image_size=256)
    restored = denormalize_image(tensor)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "roundtrip.png"
    restored.save(output_path)

    saved = Image.open(output_path)
    if saved.mode != "RGB":
        raise SystemExit(f"expected RGB, got {saved.mode}")
    if saved.size != (256, 256):
        raise SystemExit(f"expected 256x256, got {saved.size}")

    original = TF.to_tensor(resized)
    result = TF.to_tensor(saved)
    max_diff = (original - result).abs().max().item() * 255
    if max_diff > 1:
        raise SystemExit(f"round trip max difference is {max_diff:.3f} grey levels")

    print(f"shape {tuple(tensor.shape)}")
    print(f"dtype {tensor.dtype}")
    print(f"device {tensor.device}")
    print(f"saved {output_path}")
    print(f"max grey-level difference {max_diff:.3f}")


if __name__ == "__main__":
    main()
