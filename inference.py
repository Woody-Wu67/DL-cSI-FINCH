"""Run the complete two-stage DL-cSI-FINCH neural reconstruction pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from models import DualDecoderUNet, MambaDualDecoderUNet
from utils.image_io import (
    IMAGE_SUFFIXES,
    ensure_unique_names,
    list_images,
    load_grayscale_tensor,
    min_max_normalize,
    quantize_unit_tensor,
    save_tensor_image,
)
from utils.metrics import psnr, ssim
from utils.optics import reconstruct_finch_magnitude


PROJECT_ROOT = Path(__file__).resolve().parent
STAGE2_PHASES = (0, 120, 240)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch execution device."""
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def _safe_torch_load(checkpoint_path: Path, device: torch.device) -> Any:
    """Load tensor-only checkpoints while remaining compatible with older PyTorch."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a released checkpoint and require an exact architecture match."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = _safe_torch_load(checkpoint_path, device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def _find_reference(folder: Path, stem: str) -> Path | None:
    """Find a reference image by filename stem, independent of its extension."""
    if not folder.is_dir():
        return None
    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem == stem
    ]
    if len(candidates) > 1:
        raise ValueError(f"Multiple reference images share the stem '{stem}' in {folder}")
    return candidates[0] if candidates else None


def _record_metric(
    records: list[dict[str, object]],
    prediction: torch.Tensor,
    reference_path: Path | None,
    sample: str,
    stage: str,
    network: str,
    output: str,
    image_size: int,
) -> None:
    """Append PSNR and SSIM when the corresponding reference image exists."""
    if reference_path is None:
        return
    reference = load_grayscale_tensor(reference_path, size=image_size)
    prediction = prediction.detach().cpu().float().clamp(0.0, 1.0)
    records.append(
        {
            "sample": sample,
            "stage": stage,
            "network": network,
            "output": output,
            "psnr_db": psnr(prediction, reference),
            "ssim": ssim(prediction, reference),
        }
    )


def run_stage1(
    input_paths: list[Path],
    output_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
    wavelength_mm: float,
    pixel_size_mm: float,
    distance_mm: float,
    ground_truth_dir: Path | None = None,
    metric_records: list[dict[str, object]] | None = None,
) -> dict[str, torch.Tensor]:
    """Predict two holograms and reconstruct one FINCH magnitude image per input."""
    ensure_unique_names(input_paths)
    metric_records = metric_records if metric_records is not None else []
    model = load_checkpoint(DualDecoderUNet(), checkpoint_path, device)
    reconstructions: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        for index, input_path in enumerate(input_paths, start=1):
            input_tensor = load_grayscale_tensor(input_path, size=image_size).to(device)
            predicted_120, predicted_240 = model(input_tensor)

            output_name = f"{input_path.stem}.bmp"
            save_tensor_image(
                predicted_120,
                output_dir / "stage1" / "predicted_phase_120" / output_name,
            )
            save_tensor_image(
                predicted_240,
                output_dir / "stage1" / "predicted_phase_240" / output_name,
            )

            # The original workflow exchanges 8-bit BMP files with MATLAB.
            # Quantizing here reproduces that interface before numerical propagation.
            reconstruction = reconstruct_finch_magnitude(
                input_tensor,
                quantize_unit_tensor(predicted_120, mode="floor"),
                quantize_unit_tensor(predicted_240, mode="floor"),
                wavelength_mm=wavelength_mm,
                pixel_size_mm=pixel_size_mm,
                distance_mm=distance_mm,
            )
            reconstruction = quantize_unit_tensor(min_max_normalize(reconstruction))
            if not torch.isfinite(reconstruction).all():
                raise RuntimeError(f"Non-finite reconstruction produced for {input_path}")
            reconstruction_cpu = reconstruction.cpu()
            reconstructions[input_path.stem] = reconstruction_cpu
            save_tensor_image(
                reconstruction_cpu,
                output_dir / "stage1" / "reconstruction" / output_name,
                mode="round",
            )

            if ground_truth_dir is not None:
                _record_metric(
                    metric_records,
                    predicted_120,
                    _find_reference(ground_truth_dir / "phase1" / "UNet_gt1", input_path.stem),
                    input_path.stem,
                    "phase1",
                    "UNet",
                    "phase_120",
                    image_size,
                )
                _record_metric(
                    metric_records,
                    predicted_240,
                    _find_reference(ground_truth_dir / "phase1" / "UNet_gt2", input_path.stem),
                    input_path.stem,
                    "phase1",
                    "UNet",
                    "phase_240",
                    image_size,
                )
            print(f"Stage 1 [{index}/{len(input_paths)}]: {input_path.name}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reconstructions


def run_stage2(
    reconstructions: dict[str, torch.Tensor],
    output_dir: Path,
    checkpoint_paths: list[Path],
    device: torch.device,
    image_size: int,
    ground_truth_dir: Path | None = None,
    metric_records: list[dict[str, object]] | None = None,
) -> None:
    """Run the three second-stage subnetworks on every FINCH reconstruction."""
    if len(checkpoint_paths) != 3:
        raise ValueError("Exactly three second-stage checkpoints are required.")
    metric_records = metric_records if metric_records is not None else []

    for model_index, (phase_degrees, checkpoint_path) in enumerate(
        zip(STAGE2_PHASES, checkpoint_paths), start=1
    ):
        model = load_checkpoint(MambaDualDecoderUNet(), checkpoint_path, device)
        with torch.inference_mode():
            for sample_index, (stem, reconstruction) in enumerate(
                reconstructions.items(), start=1
            ):
                output_0, output_90 = model(reconstruction.to(device))
                output_name = f"{stem}.bmp"
                folder_0 = f"orientation_0_phase_{phase_degrees}"
                folder_90 = f"orientation_90_phase_{phase_degrees}"
                save_tensor_image(output_0, output_dir / "stage2" / folder_0 / output_name)
                save_tensor_image(output_90, output_dir / "stage2" / folder_90 / output_name)

                if ground_truth_dir is not None:
                    reference_root = ground_truth_dir / "phase2"
                    _record_metric(
                        metric_records,
                        output_0,
                        _find_reference(
                            reference_root / f"MambaUNet{model_index}_gt1", stem
                        ),
                        stem,
                        "phase2",
                        f"MambaUNet{model_index}",
                        f"orientation_0_phase_{phase_degrees}",
                        image_size,
                    )
                    _record_metric(
                        metric_records,
                        output_90,
                        _find_reference(
                            reference_root / f"MambaUNet{model_index}_gt2", stem
                        ),
                        stem,
                        "phase2",
                        f"MambaUNet{model_index}",
                        f"orientation_90_phase_{phase_degrees}",
                        image_size,
                    )
                print(
                    f"Stage 2, MambaUNet{model_index} "
                    f"[{sample_index}/{len(reconstructions)}]: {stem}"
                )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_reconstruction_directory(input_dir: Path, image_size: int) -> dict[str, torch.Tensor]:
    """Load saved first-stage reconstructions for a standalone stage-two run."""
    paths = list_images(input_dir)
    ensure_unique_names(paths)
    return {path.stem: load_grayscale_tensor(path, size=image_size) for path in paths}


def write_metrics(records: list[dict[str, object]], output_dir: Path) -> None:
    """Write per-image metrics and grouped means."""
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "stage", "network", "output", "psnr_db", "ssim"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = f"{record['stage']}/{record['network']}/{record['output']}"
        groups[key].append(record)
    summary = {
        key: {
            "count": len(values),
            "mean_psnr_db": sum(float(value["psnr_db"]) for value in values) / len(values),
            "mean_ssim": sum(float(value["ssim"]) for value in values) / len(values),
        }
        for key, values in groups.items()
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)


def write_manifest(
    output_dir: Path,
    device: torch.device,
    input_count: int,
    image_size: int,
    wavelength_mm: float,
    pixel_size_mm: float,
    distance_mm: float,
    elapsed_seconds: float,
) -> None:
    """Record the parameters and software versions used for a run."""
    manifest = {
        "input_count": input_count,
        "image_size": image_size,
        "wavelength_mm": wavelength_mm,
        "pixel_size_mm": pixel_size_mm,
        "propagation_distance_mm": distance_mm,
        "device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "elapsed_seconds": elapsed_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete two-stage DL-cSI-FINCH inference pipeline."
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ground_truth",
        help="Set to a nonexistent path to disable reference metrics.",
    )
    parser.add_argument("--device", default="auto", help="Examples: auto, cpu, cuda, cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--wavelength-mm", type=float, default=625e-6)
    parser.add_argument("--pixel-size-mm", type=float, default=3.45e-3)
    parser.add_argument("--distance-mm", type=float, default=4.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both neural stages and the physical FINCH reconstruction."""
    args = build_parser().parse_args(argv)
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be a positive multiple of 16.")
    device = resolve_device(args.device)
    input_paths = list_images(args.input_dir)
    ground_truth_dir = args.ground_truth_dir if args.ground_truth_dir.is_dir() else None
    metrics: list[dict[str, object]] = []
    start_time = time.perf_counter()

    print(f"Using device: {device}")
    reconstructions = run_stage1(
        input_paths=input_paths,
        output_dir=args.output_dir,
        checkpoint_path=args.weights_dir / "phase1_UNet.pth",
        device=device,
        image_size=args.image_size,
        wavelength_mm=args.wavelength_mm,
        pixel_size_mm=args.pixel_size_mm,
        distance_mm=args.distance_mm,
        ground_truth_dir=ground_truth_dir,
        metric_records=metrics,
    )
    run_stage2(
        reconstructions=reconstructions,
        output_dir=args.output_dir,
        checkpoint_paths=[
            args.weights_dir / f"phase2_MambaUNet{index}.pth" for index in range(1, 4)
        ],
        device=device,
        image_size=args.image_size,
        ground_truth_dir=ground_truth_dir,
        metric_records=metrics,
    )
    elapsed_seconds = time.perf_counter() - start_time
    write_metrics(metrics, args.output_dir)
    write_manifest(
        output_dir=args.output_dir,
        device=device,
        input_count=len(input_paths),
        image_size=args.image_size,
        wavelength_mm=args.wavelength_mm,
        pixel_size_mm=args.pixel_size_mm,
        distance_mm=args.distance_mm,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"Completed {len(input_paths)} image(s) in {elapsed_seconds:.2f} seconds.")
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
