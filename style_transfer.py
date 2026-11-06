import torch
import torch.nn.functional as F

from feature_extraction import extract_features, gram_matrix


CONTENT_LAYER = "conv4_2"
STYLE_LAYERS = {
    "conv1_1": 0.2,
    "conv2_1": 0.2,
    "conv3_1": 0.2,
    "conv4_1": 0.2,
    "conv5_1": 0.2,
}

# has to match preprocessing.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# starting values only, tune them after testing with real images
# beta = 1e6 assumes gram_matrix divides by C * H * W
DEFAULT_CONFIG = {
    "alpha": 1.0,
    "beta": 1e6,
    "lr": 0.02,
    "steps": 300,
    "style_weights": STYLE_LAYERS,
    "device": None,
    "log_every": 50,
    "callback": None,
}


def content_loss(generated_features, content_features):
    return F.mse_loss(generated_features[CONTENT_LAYER], content_features[CONTENT_LAYER])


def style_loss(generated_features, style_grams, style_weights):
    loss = 0.0
    for layer, weight in style_weights.items():
        generated_gram = gram_matrix(generated_features[layer])
        loss = loss + weight * F.mse_loss(generated_gram, style_grams[layer])
    return loss


def total_loss(c_loss, s_loss, alpha, beta):
    return alpha * c_loss + beta * s_loss


def get_pixel_limits(device):
    # valid range of a normalized image, so pixels stay between 0 and 1 after denormalizing
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    low = (0 - mean) / std
    high = (1 - mean) / std
    return low, high


def transfer_style(content_tensor, style_tensor, feature_extractor, config=None):
    settings = dict(DEFAULT_CONFIG)
    if config:
        settings.update(config)

    device = settings["device"] or content_tensor.device
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)
    feature_extractor = feature_extractor.to(device)

    # targets are fixed, so no gradients needed here
    with torch.no_grad():
        content_features = extract_features(content_tensor, feature_extractor)
        style_features = extract_features(style_tensor, feature_extractor)
        target_content = {CONTENT_LAYER: content_features[CONTENT_LAYER].detach()}
        style_grams = {}
        for layer in settings["style_weights"]:
            style_grams[layer] = gram_matrix(style_features[layer]).detach()

    generated = content_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=settings["lr"])
    low, high = get_pixel_limits(device)

    steps = settings["steps"]
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        generated_features = extract_features(generated, feature_extractor)
        c_loss = content_loss(generated_features, target_content)
        s_loss = style_loss(generated_features, style_grams, settings["style_weights"])
        loss = total_loss(c_loss, s_loss, settings["alpha"], settings["beta"])

        if not loss.requires_grad:
            raise RuntimeError("loss has no gradient, extract_features should not use torch.no_grad()")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(low, high)

        if settings["log_every"] and (step % settings["log_every"] == 0 or step == steps):
            print(f"step {step}/{steps}  total {loss.item():.4f}  "
                  f"content {c_loss.item():.4f}  style {s_loss.item():.6f}")

        if settings["callback"] is not None:
            settings["callback"](step, steps, loss.item())

    return generated.detach()
