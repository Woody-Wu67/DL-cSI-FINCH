"""Paired one-input/two-target dataset used for model training or evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class HoloNormalize:
    """Convert an image to a tensor and normalize each sample to [0, 1]."""

    def __call__(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(image):
            tensor = image.float()
        else:
            import numpy as np

            array = np.asarray(image, dtype=np.float32)
            tensor = torch.from_numpy(array).unsqueeze(0) / 255.0
        minimum = tensor.amin()
        maximum = tensor.amax()
        return (tensor - minimum) / (maximum - minimum + 1e-6)


class OneToTwoImageDataset(Dataset):
    """Load filename-matched grayscale inputs and two target images."""

    def __init__(
        self,
        input_folder: str | Path,
        target1_folder: str | Path,
        target2_folder: str | Path,
        transform: Callable | None = None,
    ):
        self.input_folder = Path(input_folder)
        self.target1_folder = Path(target1_folder)
        self.target2_folder = Path(target2_folder)
        self.transform = transform

        self.input_files = sorted(path.name for path in self.input_folder.iterdir() if path.is_file())
        target1_files = sorted(path.name for path in self.target1_folder.iterdir() if path.is_file())
        target2_files = sorted(path.name for path in self.target2_folder.iterdir() if path.is_file())
        if self.input_files != target1_files or self.input_files != target2_files:
            raise ValueError("Input and target folders must contain the same filenames.")

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, index: int) -> tuple[object, object, object]:
        filename = self.input_files[index]
        with Image.open(self.input_folder / filename) as image:
            input_image = image.convert("L").copy()
        with Image.open(self.target1_folder / filename) as image:
            target1 = image.convert("L").copy()
        with Image.open(self.target2_folder / filename) as image:
            target2 = image.convert("L").copy()

        if self.transform is not None:
            input_image = self.transform(input_image)
            target1 = self.transform(target1)
            target2 = self.transform(target2)
        return input_image, target1, target2


# Preserve the original public class name.
MyDataset_1to2 = OneToTwoImageDataset
