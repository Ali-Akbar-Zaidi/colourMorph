import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

# Indices in torchvision vgg19.features. Values are the ReLU outputs.
LAYER_INDICES = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
}

CONTENT_LAYER = "relu4_1"
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")

# Shapes for one 128×128 RGB image. relu4_1 is the AdaIN input.
EXPECTED_SHAPES = {
    "relu1_1": (1, 64, 128, 128),
    "relu2_1": (1, 128, 64, 64),
    "relu3_1": (1, 256, 32, 32),
    "relu4_1": (1, 512, 16, 16),
}


def load_feature_extractor(device=None):
    """Load frozen ImageNet VGG19 features through relu4_1."""
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:21]
    for index, layer in enumerate(model):
        if isinstance(layer, nn.ReLU):
            model[index] = nn.ReLU(inplace=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model


def extract_features(image_tensor, model):
    """Return relu1_1, relu2_1, relu3_1, and relu4_1.

    Gradients stay enabled so training can pass the generated image through
    VGG19 and update the decoder. Callers disable gradients for the content
    and style images themselves.
    """
    names = {index: name for name, index in LAYER_INDICES.items()}
    features = {}
    x = image_tensor
    for index, layer in enumerate(model):
        x = layer(x)
        if index in names:
            features[names[index]] = x
    return features


def main():
    from PIL import Image

    from preprocessing import preprocess_image

    image = Image.new("RGB", (128, 128), (180, 90, 40))
    tensor = preprocess_image(image, image_size=128)
    model = load_feature_extractor()

    if model.training:
        raise SystemExit("feature extractor must be in eval mode")
    if any(param.requires_grad for param in model.parameters()):
        raise SystemExit("VGG19 parameters must be frozen")
    if not isinstance(model[20], nn.ReLU):
        raise SystemExit("index 20 must be relu4_1")

    features = extract_features(tensor, model)
    if set(features) != set(LAYER_INDICES):
        raise SystemExit(f"unexpected feature keys: {sorted(features)}")

    for name, expected in EXPECTED_SHAPES.items():
        shape = tuple(features[name].shape)
        print(f"{name} {shape}")
        if shape != expected:
            raise SystemExit(f"{name} shape {shape} != {expected}")


if __name__ == "__main__":
    main()
