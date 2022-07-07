# Bilateral Normal Integration

[Xu Cao](https://hoshino042.github.io/homepage/), [Hiroaki Santo](https://sites.google.com/view/hiroaki-santo/), [Boxin Shi](http://alumni.media.mit.edu/~shiboxin/), [Fumio Okura](http://cvl.ist.osaka-u.ac.jp/user/okura/) and [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/en/member/matsushita/)

ECCV 2022

This is the official Python implementation of the ECCV 2022 work "Bilateral Normal Integration."
This work aims at **discontinuity preserving** surface reconstruction from a single surface normal map.
The proposed method works in both orthographic and perspective cases, is robust to outliers, and only has one hyperparameter in the objective function.

## Dependencies
Our implementation was tested using Python 3.7 and depends on `Numpy` and `Scipy` for numerical computation, `PyVista` for mesh IO, and `OpenCV` for image IO.
You can ensure the four packages are installed in your python environment by running:

 ```
 # cd to the `code` folder first
pip install -r requirements.txt
 ```

## Reproduce our results 
The `data` folder contains all surfaces we used in the paper.
Each normal map and its mask are put in a distinct folder.
For the normal map in the perspective case, its folder contains an extra `K.txt` recording the 3x3 camera intrinsic matrix.
Our code determines the perspective or orthographic case based on whether or not there is a `K.txt` in the normal map's folder.

To obtain the integrated surface of a specific normal map, pass the normal map's folder path to the script `bilateral_normal_integration.py`.
For example, 
```
python bilateral_normal_integration.py --path data/Fig4_reading
```
This script will save the integrated surface and discontinuity maps in the same folder.
The default parameter setting is `k=2` (the sigmoid function's sharpness), `iter=100` (the maximum iteration number of IRLS), 
and `tol=1e-5` (the stopping tolerance of IRLS).
You can change the parameter settings by running, for example, 
```
python bilateral_normal_integration.py --path data/supp_vase -k 4 --iter 100 --tol 1e-5
```
In our experiments, `tol` is consistently set as 1e-5.
For `k` and `iter`, we used the following setups:

|  surfaces                     | k | iter|
| ---- | ---- | ---- |
| Fig. 1 the thinker            | 2 | 100 |
| Fig. 4 stripes                | 2 | 100 |
| Fig. 4 reading                | 2 | 100 |
| Fig. 5 plant                  | 2 | 150 |
| Fig. 6 bunny                  | 2 | 100 |
| Fig. 7 all DiLiGenT objects   | 2 | 100 |
| Fig. 8 left                   | 4 | 100 |
| Fig. 8 right                  | 2 | 300 |
| supp vase                     | 4 | 100 |
| supp tent                     | 1 | 100 |


## Run on your normal maps
You can test our method using your normal maps.
Put the following in the same folder, and pass the folder path to the script, as abovementioned.

- `"normal_map.png"`: The RGB color-coded normal map. Check main paper's Fig. 1(a) for the coordinate system. 
We recommend saving the normal maps as 16-bit images to reduce the discretization error.
- `"mask.png"`: The integration domain should be white (1.0 or 255); the background should be black (0).
- `"K.txt"` (perspective case): the (3, 3) camera intrinsic matrix. We used `np.savetxt("K.txt", K)` to save the camera matrix into the txt file.

Reading the normal map from an RGB image inevitably introduces discretization errors, e.g., 
the n_x continuously defined in [-1, 1] can only take 256 or 65536 possible values in an 8-bit or 16-bit image, respectively.
If you want to avoid such error, you can directly call the function ``bilateral_normal_integration()`` in your code by

```
depth_map, surface, wu_map, wv_map = bilateral_normal_integration(normal_map, mask, k=2, K=None, max_iter=100, tol=1e-5)
```

We have to only provide the RGB color-coded normal maps due to the size limitation of supplementary files.
The results in the main paper are all integrated from indiscrete normal maps and do not suffer from such discretization errors.
The normal vectors are not discretized (though the integration domain is a discrete pixel domain).
