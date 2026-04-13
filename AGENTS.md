# AGENTS.md — Bilateral Normal Integration CLI Tool Spec

This file describes the CLI interface for AI agents. For human-readable documentation, see `README.md`.

## What this tool does

Bilateral Normal Integration (BiNI) reconstructs a 3D surface from a surface normal map. Given an RGB-encoded normal map and a binary mask, it outputs a triangle mesh (`.ply`), a depth map (`.npy`), and discontinuity weight maps (`.png`).

## Entry point

```
python cli.py <command> [options]
```

## Commands

### `info` — Inspect a dataset (no computation)

```
python cli.py info <path> [--json]
```

Use this first to check whether the dataset is valid and to learn its properties before running integration.

**JSON output schema:**
```json
{
  "path": "/absolute/path",
  "files": {
    "normal_map.png": true,
    "mask.png": true,
    "K.txt": false
  },
  "image_size": [H, W],
  "bit_depth": 16,
  "num_valid_pixels": 29376,
  "projection": "orthographic"
}
```

- `projection` is `"perspective"` when `K.txt` exists, `"orthographic"` otherwise.
- `image_size` is `[height, width]`.

---

### `run` — Run integration on a single dataset

```
python cli.py run <path> [-k K] [--iter N] [--tol T] [--output-dir DIR] [--json]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `path` | (required) | Path to dataset folder containing `normal_map.png` |
| `-k` | `2` | Sigmoid sharpness. Higher = sharper edges, more noise sensitive |
| `--iter` | `150` | Max IRLS iterations |
| `--tol` | `1e-4` | Convergence tolerance on relative energy change |
| `--output-dir` | same as `path` | Where to write output files |
| `--json` | off | Structured JSON to stdout; progress goes to stderr |

**JSON output schema (success):**
```json
{
  "status": "success",
  "path": "/absolute/path",
  "params": {
    "k": 2.0,
    "max_iter": 150,
    "tol": 0.0001,
    "projection": "perspective"
  },
  "stats": {
    "num_normals": 40670,
    "iterations": 13,
    "final_energy": 658.295,
    "wall_time": 0.642
  },
  "output_files": [
    "/absolute/path/mesh_k_2.ply",
    "/absolute/path/wu_k_2.png",
    "/absolute/path/wv_k_2.png",
    "/absolute/path/depth.npy",
    "/absolute/path/energy.npy"
  ]
}
```

**JSON output schema (error):**
```json
{
  "status": "error",
  "path": "/absolute/path",
  "error": "normal_map.png not found in /absolute/path"
}
```

---

### `batch` — Run integration on multiple datasets

```
python cli.py batch <path1> <path2> ... [-k K] [--iter N] [--tol T] [--json]
```

Same algorithm parameters as `run`. Processes datasets sequentially. One dataset's failure does not abort the batch.

**JSON output:** An array of `run` result objects (each with `"status": "success"` or `"status": "error"`).

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All succeeded |
| 1 | Input error (path not found, missing files) |
| 2 | Solver error or partial failure in batch |

## Dataset folder format

A valid dataset folder contains:

| File | Required | Description |
|------|----------|-------------|
| `normal_map.png` | yes | RGB normal map (8-bit or 16-bit) |
| `mask.png` | no | Binary mask (white = valid region). If absent, entire image is used |
| `K.txt` | no | 3x3 camera intrinsic matrix. If present, perspective mode is used |

## Typical agent workflow

```bash
# 1. Discover what datasets are available
ls data/

# 2. Inspect a dataset before running
python cli.py info data/Fig4_reading --json

# 3. Run integration
python cli.py run data/Fig4_reading -k 2 --json

# 4. Or batch-process multiple datasets
python cli.py batch data/Fig4_reading data/Fig5_owl data/Fig6_bunny --json
```

## Understanding the algorithm

The optimized solver (`bilateral_normal_integration_cpu.py`) uses Numba JIT and is not easy to read. To understand the core algorithm, read `bilateral_normal_integration_simple.py` instead — it is a clean reference implementation optimized for clarity, not speed.

## Available datasets

```
data/Fig1_thinker        — perspective, sculpture
data/Fig4_reading        — orthographic, synthetic face
data/Fig4_stripes        — orthographic, synthetic stripes
data/Fig5_human          — orthographic, real-world human
data/Fig5_owl            — orthographic, real-world owl
data/Fig5_plant          — orthographic, real-world plant
data/Fig6_bunny          — perspective, Stanford bunny
data/Fig7_diligent/      — perspective, 9 DiLiGenT objects:
    bear, buddha, cat, cow, goblet, harvest, pot1, pot2, reading
data/supp_tent           — orthographic, toy tent
data/supp_vase           — orthographic, toy vase (use -k 4)
data/supp_limitation2    — orthographic (use -k 4)
data/supp_limitation3    — orthographic (use --iter 300)
```
