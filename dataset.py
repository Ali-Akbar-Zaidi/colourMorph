import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from preprocessing import preprocess_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_tensor(path, image_size):
    with Image.open(path) as image:
        tensor = preprocess_image(image, image_size=image_size)
    return tensor.squeeze(0)


class StyleTransferDataset(Dataset):
    """Unpaired content photographs and style paintings.

    Each item is one content image and one independently chosen style image.
    """

    def __init__(self, content_dir, style_dir, image_size=128):
        self.content_paths = list_images(content_dir)
        self.style_paths = list_images(style_dir)
        self.image_size = image_size
        if not self.content_paths or not self.style_paths:
            raise FileNotFoundError(
                f"Need images in both {content_dir} and {style_dir}. "
                "Use about 300–500 COCO photos and 300–500 WikiArt paintings. "
                "Keep those datasets out of git."
            )

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, index):
        content = load_tensor(self.content_paths[index], self.image_size)
        style_path = self.style_paths[random.randrange(len(self.style_paths))]
        style = load_tensor(style_path, self.image_size)
        return content, style
