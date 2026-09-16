"""Tests for the explicit FINCH reconstruction operations."""

import math
import unittest

import numpy as np
import torch

from utils.image_io import min_max_normalize
from utils.optics import reconstruct_finch_magnitude, three_step_demodulation


class OpticsTests(unittest.TestCase):
    def test_three_step_formula(self) -> None:
        phase_0 = torch.ones(1, 1, 4, 4)
        phase_120 = torch.full_like(phase_0, 2.0)
        phase_240 = torch.full_like(phase_0, 3.0)
        result = three_step_demodulation(phase_0, phase_120, phase_240)

        coefficient_120 = complex(math.cos(-2 * math.pi / 3), math.sin(-2 * math.pi / 3))
        coefficient_240 = complex(math.cos(-4 * math.pi / 3), math.sin(-4 * math.pi / 3))
        expected = (
            coefficient_240 - coefficient_120
            + 2.0 * (1.0 - coefficient_240)
            + 3.0 * (coefficient_120 - 1.0)
        )
        self.assertTrue(torch.allclose(result, torch.full_like(result, expected)))

    def test_reconstruction_is_finite(self) -> None:
        generator = torch.Generator().manual_seed(7)
        images = [torch.rand(1, 1, 32, 32, generator=generator) for _ in range(3)]
        magnitude = reconstruct_finch_magnitude(*images)
        normalized = min_max_normalize(magnitude)
        self.assertEqual(tuple(normalized.shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_python_matches_matlab_formula(self) -> None:
        generator = torch.Generator().manual_seed(11)
        images = [torch.rand(1, 1, 32, 32, generator=generator) for _ in range(3)]
        actual = min_max_normalize(reconstruct_finch_magnitude(*images)).squeeze().numpy()

        phase_0, phase_120, phase_240 = [image.squeeze().numpy() for image in images]
        theta_1, theta_2, theta_3 = 0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0
        complex_hologram = (
            phase_0 * (np.exp(-1j * theta_3) - np.exp(-1j * theta_2))
            + phase_120 * (np.exp(-1j * theta_1) - np.exp(-1j * theta_3))
            + phase_240 * (np.exp(-1j * theta_2) - np.exp(-1j * theta_1))
        )
        wavelength = 625e-6
        pixel_size = 3.45e-3
        distance = 4.2
        height, width = complex_hologram.shape
        x = wavelength * (np.arange(width) - width / 2.0) / (pixel_size * width)
        y = wavelength * (np.arange(height) - height / 2.0) / (pixel_size * height)
        grid_x, grid_y = np.meshgrid(x, y)
        transfer = np.exp(
            1j
            * (2.0 * np.pi / wavelength)
            * distance
            * np.sqrt(1.0 - grid_x**2 - grid_y**2)
        )
        spectrum = np.fft.fftshift(np.fft.fft2(complex_hologram))
        expected = np.abs(np.fft.ifft2(spectrum * transfer))
        expected = (expected - expected.min()) / (expected.max() - expected.min())
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
