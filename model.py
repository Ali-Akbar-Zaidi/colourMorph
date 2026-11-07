import torch.nn as nn


class Decoder(nn.Module):
    """Map a relu4_1 tensor back to a 3-channel image.

    Three nearest upsamplings restore the spatial size: 16→32→64→128
    for a 128×128 input. The last convolution has no ReLU.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, features):
        return self.net(features)
