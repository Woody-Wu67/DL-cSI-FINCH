"""Shared-encoder, dual-decoder U-Net used in the first stage."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Apply two convolution, normalization, and ReLU operations."""

    def __init__(self, in_channels: int, out_channels: int, batchnorm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample a decoder feature map and fuse it with an encoder skip feature."""

    def __init__(self, in_channel: int, out_channel: int, batchnorm: bool = True):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.up_conv_block = ConvBlock(
            in_channels=in_channel,
            out_channels=out_channel,
            batchnorm=batchnorm,
        )

    def forward(self, decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(decoder_feature)
        x = torch.cat([x, skip_feature], dim=1)
        return self.up_conv_block(x)


class DualDecoderUNet(nn.Module):
    """Predict the 120-degree and 240-degree FINCH holograms from one input."""

    def __init__(self):
        super().__init__()
        self.n_filters = 22

        self.conv_1 = ConvBlock(1, self.n_filters)
        self.conv_2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            ConvBlock(self.n_filters, self.n_filters * 2),
        )
        self.conv_3 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            ConvBlock(self.n_filters * 2, self.n_filters * 4),
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            ConvBlock(self.n_filters * 4, self.n_filters * 8),
        )
        self.conv_5 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            ConvBlock(self.n_filters * 8, self.n_filters * 16),
        )

        self.up_1_1 = DecoderBlock(self.n_filters * 16, self.n_filters * 8)
        self.up_2_1 = DecoderBlock(self.n_filters * 8, self.n_filters * 4)
        self.up_3_1 = DecoderBlock(self.n_filters * 4, self.n_filters * 2)
        self.up_4_1 = DecoderBlock(self.n_filters * 2, self.n_filters)

        self.up_1_2 = DecoderBlock(self.n_filters * 16, self.n_filters * 8)
        self.up_2_2 = DecoderBlock(self.n_filters * 8, self.n_filters * 4)
        self.up_3_2 = DecoderBlock(self.n_filters * 4, self.n_filters * 2)
        self.up_4_2 = DecoderBlock(self.n_filters * 2, self.n_filters)

        # These layers are not used by the published two-output forward pass.
        # They are retained so that the released training checkpoint loads strictly.
        self.up_1_3 = DecoderBlock(self.n_filters * 16, self.n_filters * 8)
        self.up_2_3 = DecoderBlock(self.n_filters * 8, self.n_filters * 4)
        self.up_3_3 = DecoderBlock(self.n_filters * 4, self.n_filters * 2)
        self.up_4_3 = DecoderBlock(self.n_filters * 2, self.n_filters)

        self.conv_1_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)
        self.conv_2_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)
        self.conv_3_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected an input tensor with shape (N, 1, H, W).")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Input height and width must both be divisible by 16.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(x)

        c_1 = self.conv_1(x)
        c_2 = self.conv_2(c_1)
        c_3 = self.conv_3(c_2)
        c_4 = self.conv_4(c_3)
        c_5 = self.conv_5(c_4)

        x_4_1 = self.up_1_1(c_5, c_4)
        x_3_1 = self.up_2_1(x_4_1, c_3)
        x_2_1 = self.up_3_1(x_3_1, c_2)
        x_1_1 = self.up_4_1(x_2_1, c_1)

        x_4_2 = self.up_1_2(c_5, c_4)
        x_3_2 = self.up_2_2(x_4_2, c_3)
        x_2_2 = self.up_3_2(x_3_2, c_2)
        x_1_2 = self.up_4_2(x_2_2, c_1)

        output_1 = torch.sigmoid(self.conv_1_output(x_1_1))
        output_2 = torch.sigmoid(self.conv_2_output(x_1_2))
        return output_1, output_2


# Preserve the class name used by the original scripts.
U_Net_1to2 = DualDecoderUNet
