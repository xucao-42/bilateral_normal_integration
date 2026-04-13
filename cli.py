#!/usr/bin/env python3
"""Bilateral Normal Integration CLI.

Usage:
    python cli.py run <path> [-k 2] [--iter 150] [--tol 1e-4] [--output-dir DIR] [--json]
    python cli.py info <path> [--json]
    python cli.py batch <paths...> [-k 2] [--iter 150] [--tol 1e-4] [--json]
"""

import argparse
import contextlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_SOLVER_ERROR = 2


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints don't corrupt JSON output."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def load_dataset(path):
    """Load normal_map, mask, and K from a dataset folder.

    Returns dict with keys: normal_map, mask, K, projection, img_shape,
    bit_depth, num_normals.
    """
    import cv2
    import numpy as np

    normal_path = os.path.join(path, "normal_map.png")
    if not os.path.isfile(normal_path):
        raise FileNotFoundError(f"normal_map.png not found in {path}")

    raw = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"failed to read image: {normal_path}")
    normal_map = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    bit_depth = 16 if normal_map.dtype == np.uint16 else 8
    if bit_depth == 16:
        normal_map = normal_map / 65535 * 2 - 1
    else:
        normal_map = normal_map / 255 * 2 - 1

    mask_path = os.path.join(path, "mask.png")
    if os.path.isfile(mask_path):
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"failed to read image: {mask_path}")
        mask = raw_mask.astype(bool)
    else:
        mask = np.ones(normal_map.shape[:2], bool)

    K_path = os.path.join(path, "K.txt")
    K = np.loadtxt(K_path) if os.path.isfile(K_path) else None

    return {
        "normal_map": normal_map,
        "mask": mask,
        "K": K,
        "projection": "perspective" if K is not None else "orthographic",
        "img_shape": list(normal_map.shape[:2]),
        "bit_depth": bit_depth,
        "num_normals": int(np.sum(mask)),
    }


def inspect_dataset(path):
    """Inspect a dataset folder and return metadata without running integration."""
    import cv2
    import numpy as np

    if not os.path.isdir(path):
        raise FileNotFoundError(f"directory not found: {path}")

    files = {}
    for name in ("normal_map.png", "mask.png", "K.txt"):
        files[name] = os.path.isfile(os.path.join(path, name))

    result = {
        "path": os.path.abspath(path),
        "files": files,
        "image_size": None,
        "bit_depth": None,
        "num_valid_pixels": None,
        "projection": "orthographic",
    }

    if files["normal_map.png"]:
        img = cv2.imread(
            os.path.join(path, "normal_map.png"), cv2.IMREAD_UNCHANGED
        )
        if img is not None:
            h, w = img.shape[:2]
            result["image_size"] = [h, w]
            result["bit_depth"] = 16 if img.dtype == np.uint16 else 8

    if files["mask.png"]:
        raw_mask = cv2.imread(
            os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if raw_mask is not None:
            mask = raw_mask.astype(bool)
            result["num_valid_pixels"] = int(np.sum(mask))
    elif result["image_size"]:
        result["num_valid_pixels"] = result["image_size"][0] * result["image_size"][1]

    if files["K.txt"]:
        result["projection"] = "perspective"

    return result


def save_outputs(depth_map, surface, wu_map, wv_map, energy_list, mask, output_dir, k):
    """Save all output files. Returns list of saved file paths."""
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    k_str = f"{k:g}"
    saved = []

    mesh_path = os.path.join(output_dir, f"mesh_k_{k_str}.ply")
    surface.save(mesh_path, binary=False)
    saved.append(mesh_path)

    for name, wmap in [("wu", wu_map), ("wv", wv_map)]:
        colored = cv2.applyColorMap((255 * wmap).astype(np.uint8), cv2.COLORMAP_JET)
        colored[~mask] = 255
        p = os.path.join(output_dir, f"{name}_k_{k_str}.png")
        cv2.imwrite(p, colored)
        saved.append(p)

    depth_path = os.path.join(output_dir, "depth.npy")
    np.save(depth_path, depth_map)
    saved.append(depth_path)

    energy_path = os.path.join(output_dir, "energy.npy")
    np.save(energy_path, np.array(energy_list))
    saved.append(energy_path)

    return saved


def run_single(path, k, max_iter, tol, output_dir):
    """Run bilateral normal integration on a single dataset. Returns result dict."""
    from bilateral_normal_integration_cpu import bilateral_normal_integration

    try:
        data = load_dataset(path)
    except (FileNotFoundError, ValueError) as e:
        return {"status": "error", "path": os.path.abspath(path), "error": str(e)}

    try:
        t0 = time.time()
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
            normal_map=data["normal_map"],
            normal_mask=data["mask"],
            k=k,
            K=data["K"],
            max_iter=max_iter,
            tol=tol,
        )
        wall_time = time.time() - t0
    except Exception as e:
        return {"status": "error", "path": os.path.abspath(path), "error": str(e)}

    out_dir = output_dir or path
    try:
        saved = save_outputs(
            depth_map, surface, wu_map, wv_map, energy_list, data["mask"], out_dir, k
        )
    except Exception as e:
        return {"status": "error", "path": os.path.abspath(path), "error": str(e)}

    return {
        "status": "success",
        "path": os.path.abspath(path),
        "params": {"k": k, "max_iter": max_iter, "tol": tol, "projection": data["projection"]},
        "stats": {
            "num_normals": data["num_normals"],
            "iterations": len(energy_list),
            "final_energy": energy_list[-1] if energy_list else None,
            "wall_time": round(wall_time, 3),
        },
        "output_files": [os.path.abspath(p) for p in saved],
    }


# ── Subcommand handlers ─────────────────────────────────────────────

def cmd_info(args):
    path = args.path
    if not os.path.isdir(path):
        err = f"directory not found: {path}"
        if args.json:
            json.dump({"status": "error", "path": path, "error": err}, sys.stdout, indent=2)
            print(file=sys.stdout)
        else:
            print(f"Error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    info = inspect_dataset(path)

    if args.json:
        json.dump(info, sys.stdout, indent=2)
        print(file=sys.stdout)
        return EXIT_SUCCESS

    # human-readable
    print(f"Dataset: {info['path']}")
    print("-" * 40)
    for fname, exists in info["files"].items():
        tag = "found" if exists else "MISSING"
        extra = ""
        if fname == "normal_map.png" and exists and info["image_size"]:
            h, w = info["image_size"]
            extra = f" ({w}x{h}, {info['bit_depth']}-bit)"
        if fname == "K.txt" and exists:
            extra = " (perspective)"
        print(f"  {fname:<20s} {tag}{extra}")
    print()
    if info["num_valid_pixels"] is not None:
        print(f"  Valid pixels:  {info['num_valid_pixels']:,}")
    print(f"  Projection:    {info['projection']}")
    return EXIT_SUCCESS


def cmd_run(args):
    path = args.path
    if not os.path.isdir(path):
        err = f"directory not found: {path}"
        if args.json:
            json.dump({"status": "error", "path": path, "error": err}, sys.stdout, indent=2)
            print(file=sys.stdout)
        else:
            print(f"Error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if args.json:
        with _redirect_stdout_to_stderr():
            result = run_single(path, args.k, args.iter, args.tol, args.output_dir)
        json.dump(result, sys.stdout, indent=2)
        print(file=sys.stdout)
    else:
        result = run_single(path, args.k, args.iter, args.tol, args.output_dir)
        if result["status"] == "success":
            s = result["stats"]
            print(f"\nDone: {result['path']}")
            print(f"  iterations={s['iterations']}  energy={s['final_energy']:.4f}  time={s['wall_time']:.1f}s")
            for f in result["output_files"]:
                print(f"  -> {f}")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)

    return EXIT_SUCCESS if result["status"] == "success" else EXIT_SOLVER_ERROR


def cmd_batch(args):
    results = []
    has_error = False

    for path in args.paths:
        if not os.path.isdir(path):
            r = {"status": "error", "path": os.path.abspath(path), "error": f"directory not found: {path}"}
            results.append(r)
            has_error = True
            if not args.json:
                print(f"[SKIP] {path}: directory not found", file=sys.stderr)
            continue

        if args.json:
            with _redirect_stdout_to_stderr():
                r = run_single(path, args.k, args.iter, args.tol, None)
        else:
            r = run_single(path, args.k, args.iter, args.tol, None)

        results.append(r)
        if r["status"] != "success":
            has_error = True

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print(file=sys.stdout)
    else:
        # summary table
        print()
        header = f"{'Dataset':<30s} {'Status':<8s} {'Normals':>10s} {'Iters':>6s} {'Energy':>12s} {'Time':>8s}"
        print(header)
        print("-" * len(header))
        total_time = 0.0
        ok_count = 0
        for r in results:
            name = os.path.basename(r["path"])
            if r["status"] == "success":
                s = r["stats"]
                total_time += s["wall_time"]
                ok_count += 1
                print(f"{name:<30s} {'ok':<8s} {s['num_normals']:>10,d} {s['iterations']:>6d} {s['final_energy']:>12.4f} {s['wall_time']:>7.1f}s")
            else:
                print(f"{name:<30s} {'ERROR':<8s} {'-':>10s} {'-':>6s} {'-':>12s} {'-':>8s}")
        print("-" * len(header))
        print(f"Total: {ok_count}/{len(results)} succeeded{' ' * 38}{total_time:>7.1f}s")

    if has_error:
        return EXIT_SOLVER_ERROR
    return EXIT_SUCCESS


# ── Argument parser ──────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="bini",
        description="Bilateral Normal Integration — CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # shared algorithm args
    algo = argparse.ArgumentParser(add_help=False)
    algo.add_argument("-k", type=float, default=2,
                      help="sigmoid sharpness for discontinuity preservation (default: 2)")
    algo.add_argument("--iter", type=int, default=150,
                      help="max IRLS iterations (default: 150)")
    algo.add_argument("--tol", type=float, default=1e-4,
                      help="relative energy convergence tolerance (default: 1e-4)")

    # shared --json flag
    json_flag = argparse.ArgumentParser(add_help=False)
    json_flag.add_argument("--json", action="store_true",
                           help="output structured JSON to stdout")

    # run
    p_run = sub.add_parser("run", parents=[algo, json_flag],
                           help="run integration on a single dataset")
    p_run.add_argument("path", help="path to dataset folder")
    p_run.add_argument("--output-dir", default=None,
                       help="output directory (default: same as input path)")
    p_run.set_defaults(func=cmd_run)

    # info
    p_info = sub.add_parser("info", parents=[json_flag],
                            help="inspect dataset metadata (no computation)")
    p_info.add_argument("path", help="path to dataset folder")
    p_info.set_defaults(func=cmd_info)

    # batch
    p_batch = sub.add_parser("batch", parents=[algo, json_flag],
                             help="run integration on multiple datasets")
    p_batch.add_argument("paths", nargs="+", help="paths to dataset folders")
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
