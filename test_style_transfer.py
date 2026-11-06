import sys
import types

import torch
import torch.nn as nn
from torchvision.models import vgg19


# uses Muneeb's feature_extraction.py if it's there, otherwise a stand-in
# that returns the same dictionary format (random VGG19 weights, no download)
LAYER_INDICES = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,
    "conv5_1": 28,
}


def stub_load_feature_extractor(device=None):
    model = vgg19(weights=None).features[:29]
    for i, layer in enumerate(model):
        if isinstance(layer, nn.ReLU):
            model[i] = nn.ReLU(inplace=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model


def stub_extract_features(image_tensor, model):
    names = {index: name for name, index in LAYER_INDICES.items()}
    features = {}
    x = image_tensor
    for i, layer in enumerate(model):
        x = layer(x)
        if i in names:
            features[names[i]] = x
    return features


def stub_gram_matrix(feature_tensor):
    b, c, h, w = feature_tensor.shape
    flat = feature_tensor.view(b * c, h * w)
    return flat @ flat.t() / (c * h * w)


try:
    import feature_extraction
    USING_STUB = False
except ImportError:
    feature_extraction = types.ModuleType("feature_extraction")
    feature_extraction.load_feature_extractor = stub_load_feature_extractor
    feature_extraction.extract_features = stub_extract_features
    feature_extraction.gram_matrix = stub_gram_matrix
    sys.modules["feature_extraction"] = feature_extraction
    USING_STUB = True

import style_transfer
from style_transfer import content_loss, style_loss, total_loss, transfer_style, get_pixel_limits


def make_image(seed):
    torch.manual_seed(seed)
    low, high = get_pixel_limits("cpu")
    return low + torch.rand(1, 3, 256, 256) * (high - low)


def test_losses_are_zero_for_same_features():
    model = feature_extraction.load_feature_extractor()
    image = make_image(0)
    features = feature_extraction.extract_features(image, model)
    grams = {layer: feature_extraction.gram_matrix(features[layer]) for layer in style_transfer.STYLE_LAYERS}

    assert content_loss(features, features).item() == 0
    assert style_loss(features, grams, style_transfer.STYLE_LAYERS).item() == 0


def test_losses_are_positive_for_different_images():
    model = feature_extraction.load_feature_extractor()
    a = feature_extraction.extract_features(make_image(0), model)
    b = feature_extraction.extract_features(make_image(1), model)
    grams_b = {layer: feature_extraction.gram_matrix(b[layer]) for layer in style_transfer.STYLE_LAYERS}

    assert content_loss(a, b).item() > 0
    assert style_loss(a, grams_b, style_transfer.STYLE_LAYERS).item() > 0


def test_total_loss_uses_weights():
    c = torch.tensor(2.0)
    s = torch.tensor(3.0)
    assert total_loss(c, s, 1.0, 10.0).item() == 32.0


def test_transfer_style_output():
    model = feature_extraction.load_feature_extractor()
    content = make_image(0)
    style = make_image(1)
    content_before = content.clone()
    style_before = style.clone()

    losses = []
    config = {"steps": 15, "log_every": 0, "callback": lambda step, total, loss: losses.append(loss)}
    output = transfer_style(content, style, model, config)

    low, high = get_pixel_limits("cpu")
    assert output.shape == (1, 3, 256, 256)
    assert output.device == content.device
    assert output.dtype == torch.float32
    assert not output.requires_grad
    assert torch.isfinite(output).all()
    assert (output >= low - 1e-5).all() and (output <= high + 1e-5).all()
    assert torch.equal(content, content_before) and torch.equal(style, style_before)
    assert len(losses) == 15
    assert losses[-1] < losses[0]


def test_model_weights_stay_fixed():
    model = feature_extraction.load_feature_extractor()
    weights_before = [p.clone() for p in model.parameters()]
    transfer_style(make_image(0), make_image(1), model, {"steps": 3, "log_every": 0})
    for before, after in zip(weights_before, model.parameters()):
        assert torch.equal(before, after)


if __name__ == "__main__":
    print("using stand-in feature extractor" if USING_STUB else "using feature_extraction.py")
    tests = [
        test_losses_are_zero_for_same_features,
        test_losses_are_positive_for_different_images,
        test_total_loss_uses_weights,
        test_transfer_style_output,
        test_model_weights_stay_fixed,
    ]
    for test in tests:
        test()
        print("passed", test.__name__)
    print("all tests passed")
