"""Image input and output helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _natural_key(path: Path) -> list[object]:
    """Return a natural-sort key so that 2.bmp precedes 10.bmp."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    """List supported image files in deterministic natural order."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    images = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )
    if not images:
        raise FileNotFoundError(f"No supported images were found in: {folder}")
    return images


def load_grayscale_tensor(path: Path, size: int | None = None) -> torch.Tensor:
    """Load an image as a float tensor with shape (1, 1, H, W) in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None and image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def min_max_normalize(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize a tensor to [0, 1], returning zeros for a constant input."""
    minimum = tensor.amin()
    maximum = tensor.amax()
    dynamic_range = maximum - minimum
    if dynamic_range.item() <= epsilon:
        return torch.zeros_like(tensor)
    return (tensor - minimum) / dynamic_range


def quantize_unit_tensor(tensor: torch.Tensor, mode: str = "round") -> torch.Tensor:
    """Quantize a unit-range tensor to the same levels as an 8-bit image."""
    scaled = tensor.clamp(0.0, 1.0) * 255.0
    if mode == "round":
        quantized = torch.round(scaled)
    elif mode == "floor":
        quantized = torch.floor(scaled)
    else:
        raise ValueError("Quantization mode must be either 'round' or 'floor'.")
    return quantized / 255.0


def save_tensor_image(tensor: torch.Tensor, path: Path, mode: str = "floor") -> None:
    """Save a unit-range tensor as an 8-bit grayscale image."""
    array = tensor.detach().squeeze().cpu().clamp(0.0, 1.0).numpy()
    if mode == "round":
        encoded = np.rint(array * 255.0).astype(np.uint8)
    elif mode == "floor":
        encoded = (array * 255.0).astype(np.uint8)
    else:
        raise ValueError("Encoding mode must be either 'round' or 'floor'.")
    output = Image.fromarray(encoded, mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)


def ensure_unique_names(paths: Iterable[Path]) -> None:
    """Reject inputs whose filename stems would overwrite one another."""
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Input image filename stems must be unique.")
