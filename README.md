# DL-cSI-FINCH

Inference code and pretrained weights for the two-stage network described in:

> H. Wang, T. Liao, L. Zhong, *Single-shot super-resolution imaging via deep-learning-based composite structured-illumination Fresnel incoherent correlation holography*, GitHub (2026). <https://github.com/Woody-Wu67/DL-cSI-FINCH>

## Pipeline

The complete inference path implemented here is:

1. The first-stage shared-encoder, dual-decoder U-Net receives one measured, zero-phase composite-pattern FINCH hologram.
2. It predicts the missing `2π/3` and `4π/3` holographic phase states.
3. The measured and predicted holograms are combined by three-step phase-shifting demodulation.
4. Angular-spectrum propagation reconstructs a normalized FINCH magnitude image (`Rec. #1`).
5. The same reconstruction is passed to three independent Mamba U-Nets. Each network produces the `0°` and `90°` orientation pair at one structured-illumination phase (`0`, `2π/3`, or `4π/3`).

The repository therefore generates the six direction- and phase-resolved images required by the subsequent structured-illumination spectral reconstruction. The final SI spectral separation and recombination code is not part of the supplied network source and is not reimplemented here.

## Repository contents

- `inference.py`: complete first-stage, physical-propagation, and second-stage pipeline.
- `test1.py`: standalone first-stage prediction and FINCH reconstruction.
- `test2.py`: standalone second-stage inference from saved FINCH reconstructions.
- `models/`: U-Net and Mamba U-Net definitions.
- `utils/optics.py`: Python translation of `chonggoupro.m`.
- `matlab_reference/chonggoupro.m`: cleaned MATLAB reference with English comments.
- `weights/`: four released pretrained checkpoints.
- `CHECKSUMS.sha256`: SHA-256 checksums for the released checkpoints.
- `data/input/`: three example zero-phase holograms.
- `data/ground_truth/`: first- and second-stage reference images.
- `tests/`: checkpoint, tensor-shape, and propagation checks.

## Installation

Python 3.9 or newer is recommended. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for the local CUDA driver when GPU inference is required. CPU inference is supported but is substantially slower for the three Mamba U-Nets.

## Run the examples

From the repository root:

```bash
python inference.py
```

The command automatically uses `cuda:0` when CUDA is available and otherwise uses the CPU. To choose a device explicitly:

```bash
python inference.py --device cpu
python inference.py --device cuda:0
```

Run each stage separately when needed:

```bash
python test1.py
python test2.py
```

Custom input and output directories can be supplied as follows:

```bash
python inference.py --input-dir /path/to/input --output-dir /path/to/output
```

Input images are converted to single-channel grayscale and resized to `512 × 512`. Use `--image-size` only with a positive multiple of 16. The released checkpoints were trained for 512-pixel images.

## Propagation parameters

The default values reproduce the supplied MATLAB program:

- wavelength: `625e-6 mm` (625 nm)
- camera pixel size: `3.45e-3 mm` (3.45 μm)
- propagation distance: `4.2 mm`
- image dimensions: `512 × 512`

They can be overridden with `--wavelength-mm`, `--pixel-size-mm`, and `--distance-mm`.

## Outputs

Results are written under `outputs/`:

```text
outputs/
├── stage1/
│   ├── predicted_phase_120/
│   ├── predicted_phase_240/
│   └── reconstruction/
├── stage2/
│   ├── orientation_0_phase_0/
│   ├── orientation_90_phase_0/
│   ├── orientation_0_phase_120/
│   ├── orientation_90_phase_120/
│   ├── orientation_0_phase_240/
│   └── orientation_90_phase_240/
├── metrics.csv
├── metrics_summary.json
└── run_manifest.json
```

When matching reference files are available, `metrics.csv` and `metrics_summary.json` contain PSNR and SSIM comparisons. These metrics quantify agreement with the provided multi-frame-derived references; they are not independent spatial-resolution measurements.

## Verification

Run the built-in tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests verify that every checkpoint loads strictly into its model, both output branches have the expected dimensions, and the optical reconstruction remains finite and normalized.

## Notes for GitHub

Each checkpoint is below GitHub's 100 MB per-file limit, although Git LFS may still be preferable because the four files total approximately 144 MB. No software license has been selected in this package; the repository owners should add the intended license before declaring the project open source.
