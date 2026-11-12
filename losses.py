import torch.nn.functional as F

from adain import mean_std
from feature_extraction import CONTENT_LAYER, STYLE_LAYERS

DEFAULT_LAMBDA = 10.0
DEFAULT_LAMBDA_EDGE = 1.0


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


def edge_loss(generated, content):
    """Encourage the decoder to retain the content image's spatial edges."""
    channels = generated.shape[1]
    sobel_x = generated.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    kernel_x = sobel_x.expand(channels, 1, 3, 3)
    kernel_y = sobel_y.expand(channels, 1, 3, 3)

    generated_x = F.conv2d(generated, kernel_x, padding=1, groups=channels)
    generated_y = F.conv2d(generated, kernel_y, padding=1, groups=channels)
    content_x = F.conv2d(content, kernel_x, padding=1, groups=channels)
    content_y = F.conv2d(content, kernel_y, padding=1, groups=channels)
    return 0.5 * (
        F.l1_loss(generated_x, content_x) + F.l1_loss(generated_y, content_y)
    )


def total_loss(content, style, lambda_style=DEFAULT_LAMBDA):
    return content + lambda_style * style
