from pathlib import Path

import torch

from adain import adain_target
from feature_extraction import extract_features, load_feature_extractor
from model import Decoder
from preprocessing import IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best_decoder.pth"
PILOT_IMAGE_SIZE = 128


def normalize_rgb(rgb):
    """Map RGB in 0..1 to the ImageNet range used by VGG19."""
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb - mean) / std


def decode_to_normalized(decoder, features):
    """Run the decoder and keep the image in a valid normalized range.

    The last decoder layer has no ReLU. A sigmoid after it bounds the RGB
    values, and gradients still reach the decoder.
    """
    rgb = torch.sigmoid(decoder(features))
    return normalize_rgb(rgb)


def load_decoder(checkpoint_path, device=None):
    decoder = Decoder()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No decoder checkpoint at {path}. Train one with python train.py."
        )
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    decoder.load_state_dict(state)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    if device is not None:
        decoder = decoder.to(device)
    return decoder


def transfer_style(
    content_tensor,
    style_tensor,
    alpha=1.0,
    checkpoint_path=DEFAULT_CHECKPOINT,
    feature_extractor=None,
    decoder=None,
    device=None,
):
    """Style a normalized image in one forward pass. Returns the same shape."""
    device = device or content_tensor.device
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)
    if feature_extractor is None:
        feature_extractor = load_feature_extractor(device=device)
    else:
        feature_extractor = feature_extractor.to(device)
    if decoder is None:
        decoder = load_decoder(checkpoint_path, device=device)
    else:
        decoder = decoder.to(device)
    feature_extractor.eval()
    decoder.eval()

    with torch.no_grad():
        content_features = extract_features(content_tensor, feature_extractor)
        style_features = extract_features(style_tensor, feature_extractor)
        target = adain_target(content_features, style_features, alpha)
        generated = decode_to_normalized(decoder, target)
    return generated.detach()
