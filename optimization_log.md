# Optimization Log — bilateral_normal_integration_numpy.py

All experiments run on DiLiGenT benchmark (9 objects, max_iter=100, tol=1e-4, k=2, perspective mode).
Timing = wall time of `bilateral_normal_integration()` call.
MADE = Mean Absolute Depth Error (mm, lower is better).

---

## Baseline
**Git commit:** `1341ac9`
**Description:** scipy `cg(tol=...)` → `cg(rtol=...)` fix. No algorithmic changes.

| Object  | Time (s) | MADE (mm) |
|---------|----------|-----------|
| bear    | 2.31     | 0.334     |
| buddha  | 6.30     | 1.098     |
| cat     | 9.63     | 0.074     |
| cow     | 21.72    | 0.058     |
| goblet  | 11.66    | 9.018     |
| harvest | 13.00    | 1.838     |
| pot1    | 5.85     | 0.635     |
| pot2    | 2.99     | 0.220     |
| reading | 3.23     | 0.257     |
| **Total** | **76.69** | — |

---

## Mod 1: Precompute AT + replace W sparse matrix with array ops
**Status:** COMMITTED (`45089f9`) ✅
**Result:** Total 37.35s vs baseline 76.69s — **~51% speedup**. MADE identical.

**Changes:**
- Precompute `AT = A.T.tocsr()` outside the IRLS loop
- Replace `W = spdiags(...) ; A.T @ W @ A` with `A.multiply(w[:,None])` row-scaling
- Replace `A.T @ W @ b` with `AT @ (w * b)`
- Replace energy `(A@z-b).T @ W @ (A@z-b)` with `np.dot(r**2, w)`

| Object  | Time (s) | MADE (mm) | Δ Time |
|---------|----------|-----------|--------|
| bear    | 2.40     | 0.334     | +0.09  |
| buddha  | 6.31     | 1.098     | +0.01  |
| cat     | 2.56     | 0.074     | -7.07  |
| cow     | 1.60     | 0.058     | -20.12 |
| goblet  | 2.62     | 9.018     | -9.04  |
| harvest | 11.68    | 1.838     | -1.32  |
| pot1    | 4.89     | 0.635     | -0.96  |
| pot2    | 2.58     | 0.220     | -0.41  |
| reading | 2.71     | 0.257     | -0.52  |
| **Total** | **37.35** | — | **-39.34** |

---

## Mod 2: ILU preconditioner
**Status:** PENDING

**Changes:**
- Replace Jacobi preconditioner (`spdiags(1/diag, ...)`) with incomplete LU (`spilu`)

---

## Mod 3: Direct solver (spsolve) for first iteration
**Status:** PENDING

**Changes:**
- Use `spsolve` for the first IRLS iteration (cold start), then CG for subsequent iterations
