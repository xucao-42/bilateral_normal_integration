# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bilateral Normal Integration (BiNI) — ECCV 2022 paper implementation for discontinuity-preserving surface reconstruction from surface normal maps. Supports both orthographic and perspective camera models.

## Setup & Commands

```bash
pip install -r requirements.txt
```

**Run on a dataset:**
```bash
python bilateral_normal_integration_numpy.py --path data/Fig4_reading
python bilateral_normal_integration_numpy.py --path data/supp_vase -k 4 --iter 100 --tol 1e-5
```

**GPU-accelerated version (requires CUDA 11.3+ and CuPy):**
```bash
python bilateral_normal_integration_cupy.py --path <data_folder>
```

**Evaluate on DiLiGenT benchmark:**
```bash
python evaluation_diligent.py
```

## Architecture

Two parallel implementations of the same core algorithm:

- [bilateral_normal_integration_numpy.py](bilateral_normal_integration_numpy.py) — CPU implementation using NumPy/SciPy
- [bilateral_normal_integration_cupy.py](bilateral_normal_integration_cupy.py) — GPU implementation using CuPy (manual A.T @ W @ A optimization for ~2/3 speedup)

**Core algorithm** (`bilateral_normal_integration()`): Iteratively Reweighted Least Squares (IRLS) with a Conjugate Gradient solver (Jacobi preconditioning). Sigmoid weights (controlled by `k`) preserve discontinuities.

**Key helper functions** (both files):
- `generate_dx_dy()` — builds sparse matrices for partial derivatives
- `construct_facets_from()` — generates mesh facets from binary mask
- `map_depth_map_to_point_clouds()` — converts depth map to 3D vertices

**Outputs per run:** depth map (`.npy`), 3D mesh (`.ply`), discontinuity maps (front/back, horizontal/vertical).

## Data Format

Each dataset folder must contain:
- `normal_map.png` — RGB color-coded normal map (16-bit preferred)
- `mask.png` — binary mask (white = integration domain)
- `K.txt` (optional) — 3×3 camera intrinsic matrix; presence triggers perspective mode

## Key Hyperparameter

`-k` (default 2): sigmoid sharpness controlling discontinuity preservation. Higher k = sharper edges, more noise sensitive. Recommended values by surface type are documented in README.md.
