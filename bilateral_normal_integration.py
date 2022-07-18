from scipy.sparse import diags, coo_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
import time
import pyvista as pv


def move_left(mask):
    return np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return np.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return np.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def generate_dx_dy(mask, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive

    pixel_idx = np.zeros_like(mask, dtype=int)
    pixel_idx[mask] = np.arange(np.sum(mask))
    num_pixel = np.sum(mask)

    has_left_mask = np.logical_and(move_right(mask), mask)
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    data_term = [-1] * np.sum(has_left_mask) + [1] * np.sum(has_left_mask)
    row_idx = pixel_idx[has_left_mask]   # only the pixels having left neighbors have [-1, 1] in that row
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]))
    D_horizontal_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_right_mask) + [1] * np.sum(has_right_mask)
    row_idx = pixel_idx[has_right_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]))
    D_horizontal_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_top_mask) + [1] * np.sum(has_top_mask)
    row_idx = pixel_idx[has_top_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]))
    D_vertical_pos = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    data_term = [-1] * np.sum(has_bottom_mask) + [1] * np.sum(has_bottom_mask)
    row_idx = pixel_idx[has_bottom_mask]
    row_idx = np.tile(row_idx, 2)
    col_idx = np.concatenate((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]))
    D_vertical_neg = coo_matrix((data_term, (row_idx, col_idx)), shape=(num_pixel, num_pixel))

    return D_horizontal_pos / step_size, D_horizontal_neg / step_size, D_vertical_pos / step_size, D_vertical_neg / step_size


def construct_facets_from(mask):
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.hstack((4 * np.ones((np.sum(facet_top_left_mask), 1)),
               idx[facet_top_left_mask][:, None],
               idx[facet_bottom_left_mask][:, None],
               idx[facet_bottom_right_mask][:, None],
               idx[facet_top_right_mask][:, None])).astype(int)


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
                                 lambda1=0,
                                 depth_map=None,
                                 depth_mask=None,
                                 K=None,
                                 step_size=1,
                                 max_iter=100,
                                 tol=1e-5,
                                 cg_max_iter=500,
                                 cg_tol=1e-5):
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

    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
          f"The number of normal vectors is {np.sum(normal_mask)}.")
    # transfer the normal map from the normal coordinates to the camera coordinates
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    if K is not None:  # perspective
        H, W = normal_mask.shape

        yy, xx = np.meshgrid(range(W), range(H))
        xx = np.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        Nz_u = diags(uu * nx + vv * ny + fx * nz)
        Nz_v = diags(uu * nx + vv * ny + fy * nz)

    else:  # orthographic
        Nz_u = diags(nz)
        Nz_v = diags(nz)

    # get partial derivative matrices
    Dvp, Dvn, Dup, Dun = generate_dx_dy(normal_mask, step_size)

    A1 = Nz_u @ Dup
    A2 = Nz_u @ Dun
    A3 = Nz_v @ Dvp
    A4 = Nz_v @ Dvn

    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    # initialization
    W = 0.5 * diags(np.ones_like(b))
    z = np.zeros(np.sum(normal_mask))
    energy = (A @ z - b).T @ W @ (A @ z - b)

    tic = time.time()

    energy_list = []
    if depth_map is not None:
        m = depth_mask[normal_mask].astype(int)  # shape: (num_normals,)
        M = diags(m)
        if K is not None: # perspective
            z_prior = np.log(depth_map)[normal_mask]  # shape: (num_normals,)
        else:  # orthographic
            z_prior = depth_map[normal_mask]

    for i in range(max_iter):
        if depth_map is not None:
            temp = M @ (z_prior - z)
            temp[temp==0] = np.nan
            offset = np.nanmean(temp)
            z = z + offset
            z, _ = cg(A.T @ W @ A + lambda1 * M, A.T @ W @ b + lambda1 * M @ z_prior, x0=z, maxiter=cg_max_iter, tol=cg_tol)
        else:
            z, _ = cg(A.T @ W @ A, A.T @ W @ b, x0=z, maxiter=cg_max_iter, tol=cg_tol)
        # update weights
        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = diags(np.concatenate((wu, 1-wu, wv, 1-wv)))

        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        print(f"step {i+1}/{max_iter} energy: {energy} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()

    print(f"Total time: {toc - tic}")
    depth_map = np.ones_like(normal_mask, float) * np.nan
    depth_map[normal_mask] = z

    if K is not None:  # perspective
        depth_map = np.exp(depth_map)
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    else:  # orthographic
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    surface = pv.PolyData(vertices, facets)

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = np.ones_like(normal_mask) * np.nan
    wu_map[normal_mask] = wv

    wv_map = np.ones_like(normal_mask) * np.nan
    wv_map[normal_mask] = wu

    return depth_map, surface, wu_map, wv_map, energy_list


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
    parser.add_argument('-i', '--iter', type=np.uint, default=100)
    parser.add_argument('-t', '--tol', type=float, default=1e-5)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)

    if os.path.exists(os.path.join(arg.path, "K.txt")):
        K =np.loadtxt(os.path.join(arg.path, "K.txt"))
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map, mask, k=arg.k, K=K, max_iter=arg.iter, tol=arg.tol)
    else:
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map, mask, k=arg.k, K=None, max_iter=arg.iter, tol=arg.tol)

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




















