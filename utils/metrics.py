"""Image-quality metrics with no dependencies beyond PyTorch."""

import math

import torch
import torch.nn.functional as F


def _gaussian_window(
    window_size: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    coordinates = coordinates - window_size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Calculate a differentiable mean structural similarity index."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM inputs must have the same (N, C, H, W) shape.")

    channels = prediction.shape[1]
    real_window_size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    window = _gaussian_window(
        real_window_size,
        channels,
        prediction.dtype,
        prediction.device,
    )

    mean_prediction = F.conv2d(prediction, window, groups=channels)
    mean_target = F.conv2d(target, window, groups=channels)
    mean_prediction_squared = mean_prediction.square()
    mean_target_squared = mean_target.square()
    mean_product = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(prediction.square(), window, groups=channels)
        - mean_prediction_squared
    )
    variance_target = (
        F.conv2d(target.square(), window, groups=channels) - mean_target_squared
    )
    covariance = (
        F.conv2d(prediction * target, window, groups=channels) - mean_product
    )

    constant_1 = (0.01 * data_range) ** 2
    constant_2 = (0.03 * data_range) ** 2
    similarity = (
        (2.0 * mean_product + constant_1) * (2.0 * covariance + constant_2)
    ) / (
        (mean_prediction_squared + mean_target_squared + constant_1)
        * (variance_prediction + variance_target + constant_2)
    )
    return similarity.mean()


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Calculate the mean structural similarity index as a Python float."""
    return float(
        ssim_tensor(
            prediction,
            target,
            window_size=window_size,
            data_range=data_range,
        ).item()
    )


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """Calculate peak signal-to-noise ratio in decibels."""
    if prediction.shape != target.shape:
        raise ValueError("PSNR inputs must have identical shapes.")
    mean_squared_error = F.mse_loss(prediction, target).item()
    if mean_squared_error == 0.0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mean_squared_error)
