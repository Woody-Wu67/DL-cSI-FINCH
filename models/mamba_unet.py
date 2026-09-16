"""Mamba U-Net architecture used by the three second-stage subnetworks."""

import torch
import torch.nn.functional as F
from torch import nn


class StandardMambaBlock(nn.Module):
    """Bidirectional selective state-space block for one-dimensional sequences."""

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 3,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, dim + 2 * state_dim + dim)
        self.conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim,
        )
        self.gate_proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(-torch.abs(torch.randn(dim, state_dim)))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def selective_ssm(
        self,
        x: torch.Tensor,
        input_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, feature_dim = x.shape
        state_dim = self.state_dim

        transition = self.A.unsqueeze(0)
        delta = delta.permute(0, 2, 1)
        input_matrix = input_matrix.permute(0, 2, 1)
        output_matrix = output_matrix.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        discretized_transition = torch.exp(
            transition.unsqueeze(-1) * delta.unsqueeze(2)
        )
        discretized_input = delta.unsqueeze(2) * input_matrix.unsqueeze(1)

        hidden_state = torch.zeros(
            batch_size,
            feature_dim,
            state_dim,
            device=x.device,
            dtype=x.dtype,
        )
        y = torch.zeros(
            batch_size,
            feature_dim,
            sequence_length,
            device=x.device,
            dtype=x.dtype,
        )

        for index in range(sequence_length):
            hidden_state = (
                discretized_transition[..., index] * hidden_state
                + discretized_input[..., index] * x[..., index : index + 1]
            )
            y[..., index] = torch.sum(
                output_matrix[..., index].unsqueeze(1) * hidden_state,
                dim=-1,
            )

        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x
        return y.permute(0, 2, 1)

    def bidirectional_ssm(
        self,
        x: torch.Tensor,
        input_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        forward_output = self.selective_ssm(x, input_matrix, output_matrix, delta)

        backward_output = self.selective_ssm(
            torch.flip(x, dims=(1,)),
            torch.flip(input_matrix, dims=(1,)),
            torch.flip(output_matrix, dims=(1,)),
            torch.flip(delta, dims=(1,)),
        )
        return forward_output + torch.flip(backward_output, dims=(1,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        projection = self.in_proj(x)

        x_projection = projection[..., : self.dim]
        input_matrix = projection[..., self.dim : self.dim + self.state_dim]
        output_matrix = projection[
            ..., self.dim + self.state_dim : self.dim + 2 * self.state_dim
        ]
        delta = projection[..., self.dim + 2 * self.state_dim :]
        delta = torch.clamp(F.softplus(delta), max=1.0)

        x_projection = F.silu(
            self.conv1d(x_projection.permute(0, 2, 1))
        ).permute(0, 2, 1)
        state_output = self.bidirectional_ssm(
            x_projection,
            input_matrix,
            output_matrix,
            delta,
        )
        state_output = self.dropout(state_output)
        gate = F.silu(self.gate_proj(x))
        return self.out_proj(gate * state_output) + residual


class VisionMamba2D(nn.Module):
    """Apply Mamba blocks along horizontal and vertical image sequences."""

    def __init__(self, dim: int, state_dim: int = 16):
        super().__init__()
        self.h_mamba = StandardMambaBlock(dim=dim, state_dim=state_dim)
        self.v_mamba = StandardMambaBlock(dim=dim, state_dim=state_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, channels, height, width = x.shape

        horizontal = x.permute(0, 2, 3, 1).reshape(
            batch_size * height, width, channels
        )
        horizontal = self.h_mamba(horizontal).reshape(
            batch_size, height, width, channels
        )

        vertical = x.permute(0, 3, 2, 1).reshape(
            batch_size * width, height, channels
        )
        vertical = self.v_mamba(vertical).reshape(
            batch_size, width, height, channels
        )
        vertical = vertical.permute(0, 2, 1, 3)

        fused = (horizontal + vertical).permute(0, 3, 1, 2)
        return self.fusion(fused) + residual


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


class MambaDualDecoderUNet(nn.Module):
    """Separate one composite FINCH reconstruction into two orientations."""

    def __init__(self):
        super().__init__()
        self.n_filters = 22

        # The first three blocks were present during training but are inactive in
        # the released forward pass. Retaining them preserves checkpoint compatibility.
        self.bottleneck_mamba1 = VisionMamba2D(self.n_filters, state_dim=16)
        self.bottleneck_mamba2 = VisionMamba2D(self.n_filters * 2, state_dim=16)
        self.bottleneck_mamba3 = VisionMamba2D(self.n_filters * 4, state_dim=16)
        self.bottleneck_mamba4 = VisionMamba2D(self.n_filters * 8, state_dim=16)
        self.bottleneck_mamba5 = VisionMamba2D(self.n_filters * 16, state_dim=16)

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

        # The third decoder is dormant but is required by the released checkpoints.
        self.up_1_3 = DecoderBlock(self.n_filters * 16, self.n_filters * 8)
        self.up_2_3 = DecoderBlock(self.n_filters * 8, self.n_filters * 4)
        self.up_3_3 = DecoderBlock(self.n_filters * 4, self.n_filters * 2)
        self.up_4_3 = DecoderBlock(self.n_filters * 2, self.n_filters)

        self.conv_1_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)
        self.conv_2_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)
        self.conv_3_output = nn.Conv2d(self.n_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected an input tensor with shape (N, 1, H, W).")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Input height and width must both be divisible by 16.")

        c_1 = self.conv_1(x)
        c_2 = self.conv_2(c_1)
        c_3 = self.conv_3(c_2)
        c_4 = self.bottleneck_mamba4(self.conv_4(c_3))
        c_5 = self.bottleneck_mamba5(self.conv_5(c_4))

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
U_Net_1to2 = MambaDualDecoderUNet
