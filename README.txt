DL-cSI-FINCH
=============

Purpose
-------
This package runs the two trained neural-network stages for DL-cSI-FINCH and
connects them with the physical FINCH reconstruction translated from the
supplied MATLAB program chonggoupro.m.

Processing sequence
-------------------
1. Read one measured, zero-phase composite-pattern FINCH hologram.
2. Use phase1_UNet.pth to predict the 2*pi/3 and 4*pi/3 holograms.
3. Apply three-step phase-shifting demodulation.
4. Apply angular-spectrum diffraction propagation with the default parameters:
   wavelength = 625 nm, camera pixel size = 3.45 micrometres,
   propagation distance = 4.2 mm, and image size = 512 x 512.
5. Normalize the reconstructed FINCH magnitude image.
6. Feed that same image to phase2_MambaUNet1.pth,
   phase2_MambaUNet2.pth, and phase2_MambaUNet3.pth.
7. Save six images: two fringe orientations for each of the three phase states.

Requirements
------------
Python 3.9 or later is recommended.

Install packages with:

    pip install -r requirements.txt

Run the complete example with:

    python inference.py

Run on a selected device with:

    python inference.py --device cpu
    python inference.py --device cuda:0

Run the stages separately with:

    python test1.py
    python test2.py

The default input directory is data/input. Results are stored in outputs.
The included ground-truth files are used automatically to calculate PSNR and
SSIM. Per-image values are saved in outputs/metrics.csv, and grouped means are
saved in outputs/metrics_summary.json.

Verification
------------
Run:

    python -m unittest discover -s tests -v

Citation
--------
H. Wang, T. Liao, L. Zhong, Single-shot super-resolution imaging via
deep-learning-based composite structured-illumination Fresnel incoherent
correlation holography, GitHub (2026).
https://github.com/Woody-Wu67/DL-cSI-FINCH

Scope note
----------
This package produces the six direction- and phase-resolved images required by
the subsequent structured-illumination reconstruction. The final SI spectral
separation and recombination code was not included in the supplied network
source and is therefore outside this package.

License note
------------
No software license has been selected. The repository owners should add the
intended license before declaring the project open source.
