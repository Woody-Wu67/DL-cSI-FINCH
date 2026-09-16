"""Run the three second-stage Mamba U-Nets as a standalone program."""

import argparse
from pathlib import Path

from inference import (
    PROJECT_ROOT,
    load_reconstruction_directory,
    resolve_device,
    run_stage2,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage1" / "reconstruction",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ground_truth",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    metrics: list[dict[str, object]] = []
    run_stage2(
        reconstructions=load_reconstruction_directory(args.input_dir, args.image_size),
        output_dir=args.output_dir,
        checkpoint_paths=[
            args.weights_dir / f"phase2_MambaUNet{index}.pth" for index in range(1, 4)
        ],
        device=resolve_device(args.device),
        image_size=args.image_size,
        ground_truth_dir=args.ground_truth_dir if args.ground_truth_dir.is_dir() else None,
        metric_records=metrics,
    )
    write_metrics(metrics, args.output_dir)


if __name__ == "__main__":
    main()
