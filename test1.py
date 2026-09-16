"""Run the first-stage U-Net and FINCH propagation as a standalone program."""

import argparse
from pathlib import Path

from inference import PROJECT_ROOT, resolve_device, run_stage1, write_metrics
from utils.image_io import list_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights" / "phase1_UNet.pth",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ground_truth",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--wavelength-mm", type=float, default=625e-6)
    parser.add_argument("--pixel-size-mm", type=float, default=3.45e-3)
    parser.add_argument("--distance-mm", type=float, default=4.2)
    args = parser.parse_args()

    metrics: list[dict[str, object]] = []
    run_stage1(
        input_paths=list_images(args.input_dir),
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=resolve_device(args.device),
        image_size=args.image_size,
        wavelength_mm=args.wavelength_mm,
        pixel_size_mm=args.pixel_size_mm,
        distance_mm=args.distance_mm,
        ground_truth_dir=args.ground_truth_dir if args.ground_truth_dir.is_dir() else None,
        metric_records=metrics,
    )
    write_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
