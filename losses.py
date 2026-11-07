import torch.nn.functional as F

from adain import mean_std
from feature_extraction import CONTENT_LAYER, STYLE_LAYERS

DEFAULT_LAMBDA = 10.0


def content_loss(generated_features, target_features):
    """Compare relu4_1 of the generated image with the AdaIN tensor."""
    return F.mse_loss(generated_features[CONTENT_LAYER], target_features)


def style_loss(generated_features, style_features):
    """Match channel mean and standard deviation from relu1_1 through relu4_1."""
    loss = generated_features[CONTENT_LAYER].new_zeros(())
    for layer in STYLE_LAYERS:
        generated_mean, generated_std = mean_std(generated_features[layer])
        style_mean, style_std = mean_std(style_features[layer])
        loss = loss + F.mse_loss(generated_mean, style_mean) + F.mse_loss(generated_std, style_std)
    return loss


def total_loss(content, style, lambda_style=DEFAULT_LAMBDA):
    return content + lambda_style * style
