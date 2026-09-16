"""Training losses retained from the original implementation."""

import torch
from torch import nn

from utils.metrics import ssim_tensor


class SSIMLoss(nn.Module):
    """Return one minus structural similarity as a minimization objective."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        value = ssim_tensor(
            prediction,
            target,
            window_size=self.window_size,
            data_range=self.data_range,
        )
        return 1.0 - value


class SSIM(nn.Module):
    """Compatibility wrapper that returns structural similarity directly."""

    def __init__(self, window_size: int = 11, size_average: bool = True, val_range: float = 1.0):
        super().__init__()
        if not size_average:
            raise ValueError("The compatibility SSIM wrapper supports mean reduction only.")
        self.window_size = window_size
        self.data_range = val_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction / prediction.amax().clamp_min(1e-8)
        target = target / target.amax().clamp_min(1e-8)
        return ssim_tensor(
            prediction,
            target,
            window_size=self.window_size,
            data_range=self.data_range,
        )


class FFTLoss(nn.Module):
    """Compare independently normalized Fourier-magnitude spectra."""

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss_type must be either 'l1' or 'mse'.")

    @staticmethod
    def _normalized_spectrum(image: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1)).abs()
        minimum = spectrum.amin(dim=(-2, -1), keepdim=True)
        maximum = spectrum.amax(dim=(-2, -1), keepdim=True)
        return (spectrum - minimum) / (maximum - minimum + 1e-8)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(
            self._normalized_spectrum(prediction),
            self._normalized_spectrum(target),
        )
