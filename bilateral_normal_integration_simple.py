"""
Bilateral Normal Integration (BiNI) — readable reference implementation.

Optimised for clarity, not speed.  For fast versions see
bilateral_normal_integration_cpu.py (NumPy/Numba) or
bilateral_normal_integration_cupy.py (GPU/CuPy).

Solves  min_z  Σ_i  w_i · (A_i z − b_i)²  via IRLS, where sigmoid
weights (sharpness k) preserve depth discontinuities.
"""
__author__ = "Xu Cao <xucao.42@gmail.com>"
__copyright__ = "Copyright (C) 2022 Xu Cao"
__version__ = "2.0"

from scipy.sparse import csr_matrix, diags, spdiags, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import pyvista as pv


# ---------------------------------------------------------------------------
# Mask shift helpers  (used to identify neighbouring pixel pairs)
# ---------------------------------------------------------------------------

def move_left(mask):        return np.pad(mask, ((0,0),(0,1)), constant_values=0)[:,1:]
def move_right(mask):       return np.pad(mask, ((0,0),(1,0)), constant_values=0)[:,:-1]
def move_up(mask):          return np.pad(mask, ((0,1),(0,0)), constant_values=0)[1:,:]
def move_down(mask):        return np.pad(mask, ((1,0),(0,0)), constant_values=0)[:-1,:]
def move_up_left(mask):    return np.pad(mask, ((0,1),(0,1)), constant_values=0)[1:,1:]
def move_up_right(mask):   return np.pad(mask, ((0,1),(1,0)), constant_values=0)[1:,:-1]
def move_down_left(mask): return np.pad(mask, ((1,0),(0,1)), constant_values=0)[:-1,1:]
def move_down_right(mask):return np.pad(mask, ((1,0),(1,0)), constant_values=0)[:-1,:-1]


# ---------------------------------------------------------------------------
# Finite-difference matrices
# ---------------------------------------------------------------------------

def build_derivative_matrices(mask, nz_horizontal, nz_vertical, step_size=1):
    """Build four sparse finite-difference matrices (right/left/up/down).

    Each row encodes a single-pixel gradient constraint:
        nz_eff · (z_neighbour − z_curr) / step_size  ≈  −n_component

    Returns (D_right, D_left, D_up, D_down).
    """
    num_pixel = np.sum(mask)

    # Map each masked pixel to a compact 0-based index.
    pixel_idx = np.zeros_like(mask, dtype=int)
    pixel_idx[mask] = np.arange(num_pixel)

    # Boolean masks: pixel has a neighbour in each cardinal direction.
    has_left  = mask & move_right(mask)
    has_right = mask & move_left(mask)
    has_up    = mask & move_down(mask)
    has_down  = mask & move_up(mask)

    identity = lambda m: m

    def _make(has_mask, nz_eff, col_curr, col_nb):
        """Build one finite-difference matrix.

        Each row i (for pixels with a neighbour) has:
            col_curr(i): −nz_eff[i]     (current pixel)
            col_nb(i):   +nz_eff[i]     (neighbour pixel)
        """
        nz   = nz_eff[has_mask[mask]]
        data = np.stack([-nz / step_size, nz / step_size], -1).flatten()
        cols = np.stack((pixel_idx[col_curr(has_mask)], pixel_idx[col_nb(has_mask)]), -1).flatten()
        iptr = np.concatenate([[0], np.cumsum(has_mask[mask].astype(int) * 2)])
        return csr_matrix((data, cols, iptr), shape=(num_pixel, num_pixel))

    D_right = _make(has_right, nz_horizontal, identity,   move_right)  # z[right] − z[curr]
    D_left  = _make(has_left,  nz_horizontal, move_left,  identity)    # z[curr]  − z[left]
    D_up    = _make(has_up,    nz_vertical,   identity,   move_up)     # z[up]    − z[curr]
    D_down  = _make(has_down,  nz_vertical,   move_down,  identity)    # z[curr]  − z[down]

    return D_right, D_left, D_up, D_down


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def construct_facets_from(mask):
    """Return quad facets (pyvista format) from a binary mask."""
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # A facet exists where all four corners are inside the mask.
    tl_mask = mask & move_up(mask) & move_left(mask) & move_up_left(mask)

    return np.stack((
        4 * np.ones(np.sum(tl_mask), dtype=int),
        idx[tl_mask],
        idx[move_down(tl_mask)],
        idx[move_down_right(tl_mask)],
        idx[move_right(tl_mask)],
    ), axis=-1)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    """Unproject depth map to 3-D vertices in OpenCV camera coordinates
    (x right, y down, z into scene).

    If K is None  → orthographic:  vertex = (col, row, depth) * step_size
    If K is given → perspective:   vertex = K⁻¹ · [col, row, 1]ᵀ · depth
    """
    H, W = mask.shape
    col, row = np.meshgrid(range(W), range(H))   # col: x (right), row: y (down)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = col * step_size
        vertices[..., 1] = row * step_size
        vertices[..., 2] = depth_map
        return vertices[mask]
    else:
        pts = np.stack([col[mask], row[mask], np.ones(np.sum(mask))], axis=0)  # 3 × m
        return (np.linalg.inv(K) @ pts).T * depth_map[mask, np.newaxis]        # m × 3


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def bilateral_normal_integration(normal_map,
                                 normal_mask,
                                 k=2,
                                 depth_map=None,
                                 depth_mask=None,
                                 lambda1=0,
                                 K=None,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3):
    """Bilateral Normal Integration.

    Parameters
    ----------
    normal_map : (H, W, 3) float array
        Surface normal map in OpenGL convention (x right, y up, z toward viewer),
        values in [−1, 1].  Channels: red=x, green=y, blue=z.
    normal_mask : (H, W) bool array
        Integration domain.
    k : float
        Sigmoid sharpness for discontinuity preservation.
        k=0 → smooth (no discontinuities).  k=2 is a good default.
    depth_map : (H, W) float array, optional
        Prior depth map to guide integration.
    depth_mask : (H, W) bool array, optional
        Valid pixels in depth_map.
    lambda1 : float
        Depth prior weight (only used when depth_map is provided).
    K : (3, 3) float array, optional
        Camera intrinsic matrix in standard OpenCV format:
          K = [[fx,  0, cx],    cx = principal point column (≈ W/2)
               [ 0, fy, cy],    cy = principal point row    (≈ H/2)
               [ 0,  0,  1]]
        Omit for orthographic camera.
    step_size : float
        Physical pixel size (orthographic only).
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence threshold on relative energy change.
    cg_max_iter : int
        Max CG iterations per IRLS step.
    cg_tol : float
        CG residual tolerance.

    Returns
    -------
    depth_map : (H, W) float array  — NaN outside the mask
    surface   : pyvista PolyData mesh
    wu_map    : (H, W) horizontal discontinuity weight map
    wv_map    : (H, W) vertical   discontinuity weight map
    energy_list : list of per-iteration energy values
    """
    # ------------------------------------------------------------------
    # Coordinate systems
    #
    # Input normal map (OpenGL / normal coords):
    #   red channel  = x (right)
    #   green channel = y (up)
    #   blue channel  = z (toward viewer)
    #
    # Internal / output (OpenCV camera coords):
    #   x right  = column direction
    #   y down   = row direction
    #   z into scene
    #
    # K matrix must be in standard OpenCV format (cx ≈ W/2, cy ≈ H/2).
    # ------------------------------------------------------------------

    num_normals = np.sum(normal_mask)
    projection  = "perspective" if K is not None else "orthographic"
    print(f"Running BiNI: k={k}, {projection}, {num_normals} normals.")

    # Map normal channels to algorithm components:
    #   nx = normal component in the UP direction   (constrains vertical gradient)
    #   ny = normal component in the RIGHT direction (constrains horizontal gradient)
    #   nz = normal component into scene
    nx = normal_map[normal_mask, 1]     # green = up
    ny = normal_map[normal_mask, 0]     # red   = right
    nz = -normal_map[normal_mask, 2]    # blue negated = into scene

    # ------------------------------------------------------------------
    # Camera-model-dependent effective nz
    #
    # For perspective cameras the "effective nz" that scales the depth
    # gradient depends on the pixel position (u,v) and focal lengths:
    #
    #   nz_eff_vertical   (row direction) = uu·nx + vv·ny + fy·nz
    #   nz_eff_horizontal (col direction) = uu·nx + vv·ny + fx·nz
    #
    # where uu = cy − row  (signed vertical distance from principal point)
    #       vv = col − cx  (signed horizontal distance from principal point)
    #
    # For orthographic cameras, nz_eff = nz everywhere.
    # ------------------------------------------------------------------
    if K is not None:
        H, W = normal_mask.shape
        col, row = np.meshgrid(range(W), range(H))
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        uu = (cy - row)[normal_mask]   # positive above principal point
        vv = (col - cx)[normal_mask]   # positive right of principal point

        nz_vertical   = uu * nx + vv * ny + fy * nz
        nz_horizontal = uu * nx + vv * ny + fx * nz
    else:
        nz_vertical = nz_horizontal = nz

    # ------------------------------------------------------------------
    # Build the linear system  A z ≈ b
    #
    # A is a (4n × n) sparse matrix stacking four finite-difference
    # operators (up/down/right/left pixel pairs).  Each row encodes:
    #   nz_eff · (z_j − z_i) / step  ≈  −n_component
    #
    # b layout matches A:  D_up/D_down use nx,  D_right/D_left use ny.
    # ------------------------------------------------------------------
    D_right, D_left, D_up, D_down = build_derivative_matrices(
        normal_mask,
        nz_horizontal=nz_horizontal,
        nz_vertical=nz_vertical,
        step_size=step_size,
    )

    # A z ≈ b encodes 4 types of neighbour constraints (up/down/right/left)
    A = vstack([D_up, D_down, D_right, D_left])
    b = np.concatenate([-nx, -nx, -ny, -ny])

    # ------------------------------------------------------------------
    # IRLS optimisation loop
    #
    # At each iteration:
    #   1. Fix weights w  → solve the weighted normal equations:
    #          (AᵀWA) z = Aᵀ(Wb)   via Jacobi-preconditioned CG
    #   2. Fix z  → update weights via sigmoid of the gradient contrast
    # ------------------------------------------------------------------
    w      = 0.5 * np.ones(4 * num_normals)   # uniform weights at start
    z      = np.zeros(num_normals)
    energy = 0.5 * np.dot(b, b)               # initial energy (w=0.5 everywhere)

    if depth_map is not None:
        m       = depth_mask[normal_mask].astype(int)
        M       = spdiags(m, 0, num_normals, num_normals, format="csr")
        z_prior = (np.log(depth_map) if K is not None else depth_map)[normal_mask]

    energy_list = []
    tic = time.time()

    for i in tqdm(range(max_iter), desc="IRLS"):
        # ---- Step 1: solve (AᵀWA) z = Aᵀ(Wb) -------------------------
        W     = diags(w)                        # diagonal weight matrix
        A_mat = (A.T @ W @ A).tocsr()           # weighted normal equations: AᵀWA
        b_vec = A.T @ (W @ b)                   # weighted RHS: AᵀWb

        if depth_map is not None:
            # Align z to the depth prior before adding the regulariser.
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff == 0] = np.nan
            z += np.nanmean(depth_diff)

            A_mat = A_mat + lambda1 * M
            b_vec = b_vec + lambda1 * (M @ z_prior)

        D = spdiags(1 / np.clip(A_mat.diagonal(), 1e-5, None),
                    0, num_normals, num_normals)          # Jacobi preconditioner
        z, _ = cg(A_mat, b_vec, x0=z, M=D,
                  maxiter=cg_max_iter, rtol=cg_tol)

        # ---- Step 2: update weights -----------------------------------
        # Compute depth gradients in both directions for each axis
        grad_up,    grad_down  = D_up  @ z, D_down  @ z   # vertical
        grad_right, grad_left  = D_right @ z, D_left @ z   # horizontal

        # Sigmoid weight: wu → 1 when downward gradient dominates (edge above)
        #                 wu → 0 when upward gradient dominates   (edge below)
        #                 wu ≈ 0.5 at smooth regions (both sides agree)
        wu = 1 / (1 + np.exp(-k * (grad_down**2  - grad_up**2)))
        wv = 1 / (1 + np.exp(-k * (grad_left**2  - grad_right**2)))
        w  = np.concatenate([wu, 1 - wu, wv, 1 - wv])

        # Convergence: stop when relative energy change falls below tol
        energy_old = energy
        residual   = A @ z - b
        energy     = float(residual @ (w * residual))
        energy_list.append(energy)
        if abs(energy - energy_old) / energy_old < tol:
            break

    print(f"Total time: {time.time() - tic:.3f} sec")

    # ------------------------------------------------------------------
    # Reconstruct depth map and mesh
    # ------------------------------------------------------------------
    depth_map = np.full(normal_mask.shape, np.nan)
    depth_map[normal_mask] = z
    if K is not None:
        depth_map = np.exp(depth_map)   # log-depth → metric depth

    vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K, step_size=step_size)
    facets   = construct_facets_from(normal_mask)
    if normal_map[:, :, -1].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]   # flip winding if normals face away
    surface = pv.PolyData(vertices, facets)

    # wu/wv discontinuity maps (note: paper's wu = horizontal, wv = vertical,
    # which corresponds to wv/wu from the IRLS above)
    wu_map = np.full(normal_mask.shape, np.nan)
    wu_map[normal_mask] = wv
    wv_map = np.full(normal_mask.shape, np.nan)
    wv_map[normal_mask] = wu

    return depth_map, surface, wu_map, wv_map, energy_list


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import cv2
    import argparse
    import os
    import warnings
    warnings.filterwarnings('ignore')

    def dir_path(s):
        if os.path.isdir(s):
            return s
        raise FileNotFoundError(s)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=np.uint, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(
        cv2.imread(os.path.join(arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_RGB2BGR)
    normal_map = (normal_map / 65535 * 2 - 1 if normal_map.dtype == np.uint16
                  else normal_map / 255 * 2 - 1)

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"),
                          cv2.IMREAD_GRAYSCALE).astype(bool)
    except Exception:
        mask = np.ones(normal_map.shape[:2], bool)

    K_path = os.path.join(arg.path, "K.txt")
    K = np.loadtxt(K_path) if os.path.exists(K_path) else None

    depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
        normal_map=normal_map, normal_mask=mask, k=arg.k, K=K,
        max_iter=arg.iter, tol=arg.tol)

    np.save(os.path.join(arg.path, "energy"), np.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"Saved results to {arg.path}")
