from feature_extraction import CONTENT_LAYER


def mean_std(features, eps=1e-5):
    """Per-image, per-channel mean and standard deviation. Shape [B, C, 1, 1]."""
    batch, channels = features.shape[:2]
    flat = features.view(batch, channels, -1)
    mean = flat.mean(dim=2).view(batch, channels, 1, 1)
    std = flat.var(dim=2, unbiased=False).add(eps).sqrt().view(batch, channels, 1, 1)
    return mean, std


def adain(content_features, style_features):
    """Match each content channel to the style channel's mean and std."""
    content_mean, content_std = mean_std(content_features)
    style_mean, style_std = mean_std(style_features)
    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean


def adaptive_instance_norm(content_features, style_features, alpha=1.0):
    """Blend AdaIN with the original content features. alpha is 0 to 1."""
    strength = float(alpha)
    transformed = adain(content_features, style_features)
    if strength == 1.0:
        return transformed
    if strength == 0.0:
        return content_features
    return strength * transformed + (1.0 - strength) * content_features


def adain_target(content_features, style_features, alpha=1.0):
    """AdaIN at relu4_1. The result is the decoder input and content-loss target."""
    return adaptive_instance_norm(
        content_features[CONTENT_LAYER],
        style_features[CONTENT_LAYER],
        alpha,
    )
