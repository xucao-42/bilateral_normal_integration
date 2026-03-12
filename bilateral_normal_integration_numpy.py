"""
Bilateral Normal Integration (BiNI)
"""
__author__ = "Xu Cao <xucao.42@gmail.com>"
__copyright__ = "Copyright (C) 2022 Xu Cao"
__version__ = "2.0"

from scipy.sparse import spdiags, csr_matrix, vstack
import numpy as np
from tqdm.auto import tqdm
import time
import math
import pyvista as pv
import numba


@numba.njit(cache=True)
def _build_C_csr(AT_data, AT_indices, AT_indptr, Amat_indices, Amat_indptr, n_rows, nnz_Amat):
    """Build C in CSR format directly: C[k, l] = A[l, row_k] * A[l, col_k].
    Uses AT (A^T in CSR = A in CSC) to look up A column entries.
    No COO intermediate and no sort needed — ~2× faster than _build_C_coo + COO→CSR."""
    C_data = np.empty(nnz_Amat * 4, dtype=np.float64)
    C_indices = np.empty(nnz_Amat * 4, dtype=np.int32)
    C_indptr = np.empty(nnz_Amat + 1, dtype=np.int32)
    C_indptr[0] = 0
    out = 0
    for i in range(n_rows):
        for k in range(Amat_indptr[i], Amat_indptr[i + 1]):
            j = Amat_indices[k]
            pi = AT_indptr[i]; end_i = AT_indptr[i + 1]
            pj = AT_indptr[j]; end_j = AT_indptr[j + 1]
            while pi < end_i and pj < end_j:
                li = AT_indices[pi]; lj = AT_indices[pj]
                if li == lj:
                    C_data[out] = AT_data[pi] * AT_data[pj]
                    C_indices[out] = li
                    out += 1; pi += 1; pj += 1
                elif li < lj:
                    pi += 1
                else:
                    pj += 1
            C_indptr[k + 1] = out
    return C_data[:out], C_indices[:out], C_indptr


@numba.njit(cache=True, fastmath=True)
def _pcg_jacobi(data, indices, indptr, b, d_inv, x0, max_iter, tol):
    """Jacobi-preconditioned CG, JIT-compiled to avoid Python loop overhead.
    Pre-allocates all buffers; avoids malloc per CG iteration."""
    n = len(b)
    x = x0.copy()
    r = np.empty(n)
    z = np.empty(n)
    p = np.empty(n)
    Ap = np.empty(n)

    # r = b - A @ x
    for i in range(n):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        r[i] = b[i] - s

    # z = D^{-1} r, p = z, rz = r·z, tol check
    b_norm_sq = 0.0
    rz = 0.0
    for i in range(n):
        b_norm_sq += b[i] * b[i]
        z[i] = d_inv[i] * r[i]
        p[i] = z[i]
        rz += r[i] * z[i]
    tol_sq = tol * tol * b_norm_sq

    for _ in range(max_iter):
        # Ap = A @ p
        pAp = 0.0
        for i in range(n):
            s = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                s += data[j] * p[indices[j]]
            Ap[i] = s
            pAp += p[i] * s
        if pAp == 0.0:
            break
        alpha = rz / pAp
        r_norm_sq = 0.0
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            r_norm_sq += r[i] * r[i]
        if r_norm_sq <= tol_sq:
            break
        rz_new = 0.0
        for i in range(n):
            z[i] = d_inv[i] * r[i]
            rz_new += r[i] * z[i]
        beta = rz_new / rz
        for i in range(n):
            p[i] = z[i] + beta * p[i]
        rz = rz_new
    return x


@numba.njit(cache=True, fastmath=True)
def _spmatvec_inplace(data, indices, indptr, x, out):
    """Compute out[:] = sparse_matrix @ x in-place (no allocation)."""
    for i in range(len(out)):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        out[i] = s


@numba.njit(cache=True)
def _update_weights_energy(A1z, A2z, A3z, A4z, nx, ny, k, w_out):
    """Fused weight update + energy: writes w_out in-place, returns scalar energy.
    w_out layout: [wu, 1-wu, wv, 1-wv]. Residuals use b=(-nx,-nx,-ny,-ny)."""
    n = len(A1z)
    energy = 0.0
    for i in range(n):
        d_u = A2z[i] * A2z[i] - A1z[i] * A1z[i]
        d_v = A4z[i] * A4z[i] - A3z[i] * A3z[i]
        # Numerically stable sigmoid (avoids exp overflow)
        if d_u >= 0.0:
            wu_i = 1.0 / (1.0 + math.exp(-k * d_u))
        else:
            e = math.exp(k * d_u)
            wu_i = e / (1.0 + e)
        if d_v >= 0.0:
            wv_i = 1.0 / (1.0 + math.exp(-k * d_v))
        else:
            e = math.exp(k * d_v)
            wv_i = e / (1.0 + e)
        w_out[i] = wu_i
        w_out[n + i] = 1.0 - wu_i
        w_out[2 * n + i] = wv_i
        w_out[3 * n + i] = 1.0 - wv_i
        r1 = A1z[i] + nx[i]
        r2 = A2z[i] + nx[i]
        r3 = A3z[i] + ny[i]
        r4 = A4z[i] + ny[i]
        energy += r1*r1*wu_i + r2*r2*(1.0-wu_i) + r3*r3*wv_i + r4*r4*(1.0-wv_i)
    return energy


# Define helper functions for moving masks in different directions
def move_left(mask):         return np.pad(mask, ((0,0),(0,1)), constant_values=0)[:,1:]
def move_right(mask):        return np.pad(mask, ((0,0),(1,0)), constant_values=0)[:,:-1]
def move_top(mask):          return np.pad(mask, ((0,1),(0,0)), constant_values=0)[1:,:]
def move_bottom(mask):       return np.pad(mask, ((1,0),(0,0)), constant_values=0)[:-1,:]
def move_top_left(mask):     return np.pad(mask, ((0,1),(0,1)), constant_values=0)[1:,1:]
def move_top_right(mask):    return np.pad(mask, ((0,1),(1,0)), constant_values=0)[1:,:-1]
def move_bottom_left(mask):  return np.pad(mask, ((1,0),(0,1)), constant_values=0)[:-1,1:]
def move_bottom_right(mask): return np.pad(mask, ((1,0),(1,0)), constant_values=0)[:-1,:-1]


def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive
    num_pixel = np.sum(mask)

    # Generate an integer index array with the same shape as the mask.
    pixel_idx = np.zeros_like(mask, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[mask] = np.arange(num_pixel)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    nz_left = nz_horizontal[has_left_mask[mask]]
    nz_right = nz_horizontal[has_right_mask[mask]]
    nz_top = nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    # Initialize an array 'idx' of the same shape as 'mask' with integers
    # representing the indices of valid pixels in the mask.
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # Generate masks for neighboring pixels to define facets
    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    # Identify the top-left pixel of each facet by performing a logical AND operation
    # on the masks of neighboring pixels and the input mask.
    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

    # Create masks for the other three vertices of each facet by shifting the top-left mask.
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    # Return a numpy array of facets by stacking the indices of the four vertices
    # of each facet along the last dimension. Each row of the resulting array represents
    # a single facet with the format [4, idx_top_left, idx_bottom_left, idx_bottom_right, idx_top_right].
    return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
               idx[facet_top_left_mask],
               idx[facet_bottom_left_mask],
               idx[facet_bottom_right_mask],
               idx[facet_top_right_mask]), axis=-1).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]  # m x 3

    return vertices


def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))


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
    """
    This function performs the bilateral normal integration algorithm, as described in the paper.
    It takes as input the normal map, normal mask, and several optional parameters to control the integration process.

    :param normal_map: A normal map, which is an image where each pixel's color encodes the corresponding 3D surface normal.
    :param normal_mask: A binary mask that indicates the region of interest in the normal_map to be integrated.
    :param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.

    :param depth_map: (Optional) An initial depth map to guide the integration process.
    :param depth_mask: (Optional) A binary mask that indicates the valid depths in the depth_map.

    :param lambda1 (Optional): A regularization parameter that controls the influence of the depth_map on the final result.
                               Required when depth map is input.
                               The larger the lambda1 is, the result more close to the initial depth map (fine details from the normal map are less reflected)

    :param K: (Optional) A 3x3 camera intrinsic matrix, used for perspective camera models. If not provided, the algorithm assumes an orthographic camera model.
    :param step_size: (Optional) The pixel size in the world coordinates. Default value is 1.
                                 Used only in the orthographic camera mdoel.
                                 Default value should be fine, unless you know the true value of the pixel size in the world coordinates.
                                 Do not adjust it in perspective camera model.

    :param max_iter: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.

    :return: depth_map: The resulting depth map after the bilateral normal integration process.
             surface: A pyvista PolyData mesh representing the 3D surface reconstructed from the depth map.
             wu_map: A 2D image that represents the horizontal smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             wv_map: A 2D image that represents the vertical smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             energy_list: A list of energy values during the optimization process.
    """
    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]
    # I forgot why I chose the awkward coordinate system after getting used to opencv convention :(
    # but I won't touch the working code.

    num_normals = np.sum(normal_mask)
    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
          f"The number of normal vectors is {num_normals}.")
    # Transform the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    # Handle perspective and orthographic cases separately
    if K is not None:  # perspective
        img_height, img_width = normal_mask.shape[:2]

        yy, xx = np.meshgrid(range(img_width), range(img_height))
        xx = np.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
    else:  # orthographic
        nz_u = nz_v = nz

    # get partial derivative matrices
    # right, left, top, bottom
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    # Construct the linear system
    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    # Precompute A transpose once (used every iteration)
    AT = A.T.tocsr()

    # Precompute in-place A_mat update: C s.t. A_mat.data = C @ w (avoids sparse matmul each iter)
    _A_mat_ref = (AT @ A).tocsr()
    _A_mat_ref.sort_indices()
    _A_mat_ref_indices32 = _A_mat_ref.indices.astype(np.int32)
    _A_mat_ref_indptr32 = _A_mat_ref.indptr.astype(np.int32)
    _AT_indices32 = AT.indices.astype(np.int32)
    _AT_indptr32 = AT.indptr.astype(np.int32)
    _Cv, _Ci, _Cp = _build_C_csr(
        AT.data, _AT_indices32, _AT_indptr32,
        _A_mat_ref_indices32, _A_mat_ref_indptr32, num_normals, _A_mat_ref.nnz)
    _C = csr_matrix((_Cv, _Ci, _Cp), shape=(_A_mat_ref.nnz, 4 * num_normals))
    _rows_k = np.repeat(np.arange(num_normals, dtype=np.int32), np.diff(_A_mat_ref.indptr))
    _diag_pos = np.flatnonzero(_rows_k == _A_mat_ref.indices)
    _AT_b = AT.copy()
    _AT_b.data = _AT_b.data * b[_AT_b.indices]
    _AT_b.indices = _AT_b.indices.astype(np.int32)
    _AT_b.indptr = _AT_b.indptr.astype(np.int32)
    _b_vec = np.empty(num_normals)
    del _Cv, _Ci, _Cp, _rows_k, _A_mat_ref_indices32, _A_mat_ref_indptr32, _AT_indices32, _AT_indptr32

    _has_depth_prior = depth_map is not None

    # Initialize variables for the optimization process
    w = 0.5 * np.ones(4 * num_normals)
    z = np.zeros(num_normals)
    energy = 0.5 * np.dot(b, b)

    tic = time.time()

    energy_list = []
    if depth_map is not None:
        m = depth_mask[normal_mask].astype(int)
        M = spdiags(m, 0, num_normals, num_normals, format="csr")
        z_prior = np.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]

    pbar = tqdm(range(max_iter))

    # Optimization loop
    for i in pbar:
        # fix weights and solve for depths
        if depth_map is None:
            # Fast path: numba in-place SpMV + fused weight/energy kernel
            _spmatvec_inplace(_C.data, _C.indices, _C.indptr, w, _A_mat_ref.data)
            _spmatvec_inplace(_AT_b.data, _AT_b.indices, _AT_b.indptr, w, _b_vec)
            d_inv = 1.0 / np.clip(_A_mat_ref.data[_diag_pos], 1e-5, None)
            z = _pcg_jacobi(_A_mat_ref.data, _A_mat_ref.indices, _A_mat_ref.indptr,
                            _b_vec, d_inv, z, cg_max_iter, cg_tol)
            A1z = A1 @ z; A2z = A2 @ z; A3z = A3 @ z; A4z = A4 @ z
            energy_old = energy
            energy = _update_weights_energy(A1z, A2z, A3z, A4z, nx, ny, float(k), w)
        else:
            A_mat = AT @ A.multiply(w[:, np.newaxis])
            b_vec = AT @ (w * b)
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior
            d_inv = 1.0 / np.clip(A_mat.diagonal(), 1e-5, None)
            A_csr = A_mat.tocsr()
            z = _pcg_jacobi(A_csr.data, A_csr.indices, A_csr.indptr,
                            b_vec, d_inv, z, cg_max_iter, cg_tol)
            A1z = A1 @ z; A2z = A2 @ z; A3z = A3 @ z; A4z = A4 @ z
            wu = sigmoid(A2z ** 2 - A1z ** 2, k)
            wv = sigmoid(A4z ** 2 - A3z ** 2, k)
            w = np.concatenate((wu, 1-wu, wv, 1-wv))
            energy_old = energy
            r1 = A1z - b[:num_normals];  r2 = A2z - b[num_normals:2*num_normals]
            r3 = A3z - b[2*num_normals:3*num_normals]; r4 = A4z - b[3*num_normals:]
            energy = (np.dot(r1 ** 2, wu) + np.dot(r2 ** 2, 1 - wu)
                      + np.dot(r3 ** 2, wv) + np.dot(r4 ** 2, 1 - wv))
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()

    print(f"Total time: {toc - tic:.3f} sec")

    # Extract wu/wv from w for discontinuity map output (fast path stores them in w)
    if not _has_depth_prior:
        wu = w[:num_normals]
        wv = w[2 * num_normals:3 * num_normals]

    # Reconstruct the depth map and surface
    depth_map = np.full(normal_mask.shape, np.nan)
    depth_map[normal_mask] = z

    if K is not None:  # perspective
        depth_map = np.exp(depth_map)
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    else:  # orthographic
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    if normal_map[:, :, -1].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]
    surface = pv.PolyData(vertices, facets)

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = np.full(normal_mask.shape, np.nan)
    wu_map[normal_mask] = wv

    wv_map = np.full(normal_mask.shape, np.nan)
    wv_map[normal_mask] = wu

    return depth_map, surface, wu_map, wv_map, energy_list


# Pre-load numba JIT functions from cache at module import time.
# Without this, the first object processed pays the cache-load penalty (~0.15s).
_wup = np.ones(2, dtype=np.float64)
_wup_i32 = np.array([0, 1, 0, 1], dtype=np.int32)
_wup_ip32 = np.array([0, 2, 4], dtype=np.int32)
_build_C_csr(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64), _wup_i32, _wup_ip32,
             np.array([0, 1], dtype=np.int32), np.array([0, 1, 2], dtype=np.int32), 2, 2)
_pcg_jacobi(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64), _wup_i32, _wup_ip32,
            _wup, _wup, _wup, 1, 1e-3)
_spmatvec_inplace(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64), _wup_i32, _wup_ip32,
                  _wup, _wup)
_update_weights_energy(_wup, _wup, _wup, _wup, _wup, _wup, 2.0, np.zeros(8))
del _wup, _wup_i32, _wup_ip32


if __name__ == '__main__':
    import cv2
    import argparse, os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + np.exp(-k * x)).
    # This overflow issue does not affect our results as np.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=np.uint, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)

    K_path = os.path.join(arg.path, "K.txt")
    K = np.loadtxt(K_path) if os.path.exists(K_path) else None
    depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
        normal_map=normal_map, normal_mask=mask, k=arg.k, K=K,
        max_iter=arg.iter, tol=arg.tol)

    # save the resultant polygon mesh and discontinuity maps.
    np.save(os.path.join(arg.path, "energy"), np.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"saved {arg.path}")
