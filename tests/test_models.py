"""Checkpoint compatibility and output-shape tests."""

import unittest
from pathlib import Path

import torch

from inference import load_checkpoint
from models import DualDecoderUNet, MambaDualDecoderUNet


ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_stage1_checkpoint_and_shape(self) -> None:
        model = load_checkpoint(
            DualDecoderUNet(), ROOT / "weights" / "phase1_UNet.pth", torch.device("cpu")
        )
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 1, 32, 32))
        self.assertEqual(tuple(outputs[0].shape), (1, 1, 32, 32))
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 32, 32))

    def test_stage2_checkpoints_and_shape(self) -> None:
        for index in range(1, 4):
            model = load_checkpoint(
                MambaDualDecoderUNet(),
                ROOT / "weights" / f"phase2_MambaUNet{index}.pth",
                torch.device("cpu"),
            )
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 1, 32, 32))
            self.assertEqual(tuple(outputs[0].shape), (1, 1, 32, 32))
            self.assertEqual(tuple(outputs[1].shape), (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
