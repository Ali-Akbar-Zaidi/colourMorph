import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

# Indices in torchvision vgg19.features. Values are the conv outputs at these layers.
LAYER_INDICES = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,
    "conv5_1": 28,
}

EXPECTED_SHAPES = {
    "conv1_1": (1, 64, 256, 256),
    "conv2_1": (1, 128, 128, 128),
    "conv3_1": (1, 256, 64, 64),
    "conv4_1": (1, 512, 32, 32),
    "conv4_2": (1, 512, 32, 32),
    "conv5_1": (1, 512, 16, 16),
}


def load_feature_extractor(device=None):
    """Load pretrained VGG19 features through conv5_1, frozen and in eval mode."""
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]
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
    """Return activations for the agreed content and style layers.

    Gradients stay enabled. The style-transfer loop disables them itself
    when it caches the content and style targets.
    """
    names = {index: name for name, index in LAYER_INDICES.items()}
    features = {}
    x = image_tensor
    for index, layer in enumerate(model):
        x = layer(x)
        if index in names:
            features[names[index]] = x
    return features


def gram_matrix(feature_tensor):
    """Channel correlation matrix, divided by C * H * W."""
    _, channels, height, width = feature_tensor.shape
    flattened = feature_tensor.reshape(channels, height * width)
    return flattened @ flattened.transpose(0, 1) / (channels * height * width)


def main():
    from PIL import Image

    from preprocessing import preprocess_image

    image = Image.new("RGB", (256, 256), (180, 90, 40))
    tensor = preprocess_image(image, image_size=256)
    model = load_feature_extractor()

    if model.training:
        raise SystemExit("feature extractor must be in eval mode")
    if any(param.requires_grad for param in model.parameters()):
        raise SystemExit("VGG19 parameters must be frozen")

    features = extract_features(tensor, model)
    if set(features) != set(LAYER_INDICES):
        raise SystemExit(f"unexpected feature keys: {sorted(features)}")

    for name, expected in EXPECTED_SHAPES.items():
        shape = tuple(features[name].shape)
        print(f"{name} {shape}")
        if shape != expected:
            raise SystemExit(f"{name} shape {shape} != {expected}")
        gram_shape = tuple(gram_matrix(features[name]).shape)
        print(f"{name} gram {gram_shape}")
        expected_gram = (expected[1], expected[1])
        if gram_shape != expected_gram:
            raise SystemExit(f"{name} gram {gram_shape} != {expected_gram}")


if __name__ == "__main__":
    main()
