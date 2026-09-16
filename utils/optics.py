"""Explicit three-step FINCH demodulation and numerical propagation."""

import math

import torch


def three_step_demodulation(
    phase_0: torch.Tensor,
    phase_120: torch.Tensor,
    phase_240: torch.Tensor,
) -> torch.Tensor:
    """Form the complex FINCH hologram from three 120-degree phase steps."""
    complex_dtype = (
        torch.complex128 if phase_0.dtype == torch.float64 else torch.complex64
    )
    coefficient_120 = torch.exp(
        torch.tensor(
            -1j * 2.0 * math.pi / 3.0,
            device=phase_0.device,
            dtype=complex_dtype,
        )
    )
    coefficient_240 = torch.exp(
        torch.tensor(
            -1j * 4.0 * math.pi / 3.0,
            device=phase_0.device,
            dtype=complex_dtype,
        )
    )
    return (
        phase_0 * (coefficient_240 - coefficient_120)
        + phase_120 * (1.0 - coefficient_240)
        + phase_240 * (coefficient_120 - 1.0)
    )


def angular_spectrum_propagation(
    field: torch.Tensor,
    wavelength_mm: float,
    pixel_size_mm: float,
    distance_mm: float,
) -> torch.Tensor:
    """Propagate a complex field with the angular-spectrum transfer function."""
    if field.ndim != 4:
        raise ValueError("Expected a field tensor with shape (N, C, H, W).")
    if wavelength_mm <= 0.0 or pixel_size_mm <= 0.0:
        raise ValueError("Wavelength and pixel size must be positive.")

    height, width = field.shape[-2:]
    coordinate_y = wavelength_mm * (
        torch.arange(height, device=field.device, dtype=field.real.dtype) - height / 2.0
    ) / (pixel_size_mm * height)
    coordinate_x = wavelength_mm * (
        torch.arange(width, device=field.device, dtype=field.real.dtype) - width / 2.0
    ) / (pixel_size_mm * width)
    grid_y, grid_x = torch.meshgrid(coordinate_y, coordinate_x, indexing="ij")

    radicand = 1.0 - grid_x.square() - grid_y.square()
    propagation_phase = torch.sqrt(radicand.to(field.dtype))
    wave_number = 2.0 * math.pi / wavelength_mm
    transfer = torch.exp(1j * wave_number * distance_mm * propagation_phase)

    # This order intentionally matches chonggoupro.m. Omitting ifftshift only
    # changes the alternating phase of even-sized arrays, not the final magnitude.
    centered_spectrum = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    return torch.fft.ifft2(centered_spectrum * transfer)


def reconstruct_finch_magnitude(
    phase_0: torch.Tensor,
    phase_120: torch.Tensor,
    phase_240: torch.Tensor,
    wavelength_mm: float = 625e-6,
    pixel_size_mm: float = 3.45e-3,
    distance_mm: float = 4.2,
) -> torch.Tensor:
    """Demodulate and back-propagate three phase-shifted FINCH holograms."""
    if not (phase_0.shape == phase_120.shape == phase_240.shape):
        raise ValueError("All three phase-shifted holograms must have the same shape.")
    output_dtype = phase_0.dtype
    # MATLAB converts the input images to double precision. Matching that choice
    # avoids phase errors in the rapidly varying angular-spectrum transfer kernel.
    complex_hologram = three_step_demodulation(
        phase_0.to(torch.float64),
        phase_120.to(torch.float64),
        phase_240.to(torch.float64),
    )
    propagated_field = angular_spectrum_propagation(
        complex_hologram,
        wavelength_mm=wavelength_mm,
        pixel_size_mm=pixel_size_mm,
        distance_mm=distance_mm,
    )
    return propagated_field.abs().to(output_dtype)
