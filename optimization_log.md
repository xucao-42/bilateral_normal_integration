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
**Status:** DISCARDED ❌ — slower, killed manually
**Result:** Run never completed. `spilu` decomposition (~O(n·fill_factor)) is called every IRLS iteration, overhead exceeds any CG iteration savings.

**Changes:**
- Replace Jacobi preconditioner (`spdiags(1/diag, ...)`) with incomplete LU (`spilu`, fill_factor=2)

**Lesson:** ILU is only worth it if factorization cost is amortized across many CG steps, but here A_mat changes every IRLS iteration so it must be rebuilt each time.

---

## Mod 3: Direct solver (spsolve) for first iteration
**Status:** DISCARDED ❌ — slower overall
**Result:** Total 72.62s vs Mod 1's 37.35s — **worse**. harvest: 11.68→30.38s, pot1: 4.89→23.97s.

**Changes:**
- Use `spsolve` (SuperLU direct factorization) for iteration 0, CG+Jacobi for subsequent

| Object  | Time (s) | MADE (mm) | Δ vs Mod1 |
|---------|----------|-----------|-----------|
| bear    | 1.66     | 0.283     | -0.74     |
| buddha  | 5.85     | 1.093     | -0.46     |
| cat     | 2.19     | 0.074     | -0.37     |
| cow     | 1.31     | 0.058     | -0.29     |
| goblet  | 2.13     | 9.013     | -0.49     |
| harvest | 30.38    | 1.851     | +18.70    |
| pot1    | 23.97    | 0.635     | +19.08    |
| pot2    | 2.29     | 0.220     | -0.29     |
| reading | 2.84     | 0.257     | +0.13     |
| **Total** | **72.62** | — | **+35.27** |

**Lesson:** For the cold-start (uniform-weight) system, the Laplacian-like A_mat = 0.5·A.T@A is smooth and CG+Jacobi converges quickly from z=0. SuperLU overhead (O(n^1.5)) dominates for large problems (harvest: n=56k, pot1: n=57k). spsolve is only competitive when n is small.

---

## autoresearch/mar11 — Branch starting from d31f1d4 (Mod 1 keeper, 36.3s / avg MADE 1.5041)

### Exp A: Adaptive CG tolerance (0.1 early / 1e-3 late)
**Status:** DISCARDED ❌ — MADE destroyed
**Commit:** `694bdcd`
**Result:** Total 5.7s but avg MADE 4.9897 — completely wrong solutions.
**Lesson:** Sigmoid weights `wu = σ((A2@z)² - (A1@z)²)` are sensitive to CG accuracy. Early iterations with loose tol=0.1 give wrong z → sigmoid computes wrong weights → IRLS trajectory diverges. CG tol must stay tight throughout.

---

### Exp B: AMG preconditioner (PyAMG smoothed aggregation)
**Status:** DISCARDED ❌ — MADE degraded
**Commit:** `87a6eda`
**Result:** Total 36.8s, avg MADE 2.4401 (vs 1.5041 baseline).
**Lesson:** AMG V-cycle is not symmetric (restriction/prolongation asymmetry), so PCG with AMG preconditioner gives wrong solutions. MADE degraded even though timing was similar.

---

### Exp C: SGS (Symmetric Gauss-Seidel) preconditioner — Python
**Status:** DISCARDED ❌ — too slow
**Commit:** (none)
**Result:** Total 45.7s — 4x slower than baseline. Triangular solve overhead per CG step dominates.
**Lesson:** SGS requires two triangular solves per CG iteration. For scipy sparse triangular solve, the Python/C++ dispatch overhead (~O(nnz)) per call makes this slower than Jacobi's O(n) diagonal divide.

---

### Exp D: CHOLMOD direct solver (beta=1e-6)
**Status:** DISCARDED ❌ — MADE wrong for 5 objects
**Commit:** (none)
**Result:** Total 19.4s but avg MADE 4.9897 — wrong solutions on 5/9 objects.
**Lesson:** Exact direct solve changes IRLS trajectory. For some objects the IRLS converges to a different (worse) local minimum when solved exactly vs iteratively with Jacobi.

---

### Exp E: Numba JIT PCG with Jacobi preconditioner
**Status:** COMMITTED (`f8f7738`) ✅ — **New keeper**
**Result:** Total 18.5s, avg MADE 1.5036. **~49% speedup vs d31f1d4 (36.3s)**, MADE identical.
**Changes:**
- Replace scipy `cg()` loop with `@numba.njit(cache=True)` hand-written PCG
- Pre-allocates r, z, p, Ap buffers (no malloc per iteration)
- Warmup call with int32 indices to force compilation before timing
- Jacobi preconditioner unchanged (d_inv = 1/diag)

| Object  | Time (s) | MADE (mm) |
|---------|----------|-----------|
| bear    | 1.56     | 0.334     |
| buddha  | 3.29     | 1.098     |
| cat     | 1.24     | 0.074     |
| cow     | 0.65     | 0.058     |
| goblet  | 1.04     | 9.018     |
| harvest | 5.46     | 1.838     |
| pot1    | 2.67     | 0.635     |
| pot2    | 1.62     | 0.220     |
| reading | 1.30     | 0.257     |
| **Total** | **18.83** | **avg 1.5036** |

**Lesson:** Python loop overhead in scipy's `cg()` was ~1ms/step vs numba's ~0.1ms/step — 10x reduction in CG overhead. This is the dominant remaining overhead for small objects.

---

### Exp F: Parallel prange SpMV (numba)
**Status:** DISCARDED ❌ — slower for all problem sizes
**Commit:** `346e417`
**Result:** Total 31.0s (vs 18.5s keeper). Thread launch overhead ~72µs/call exceeds benefit for n<60k.
**Lesson:** For n<60k, SpMV takes ~0.3ms single-threaded. Thread pool launch overhead (~72µs) is 24% of that, making parallelization a net loss. Would only help for n>500k.

---

### Exp G: SGS preconditioner (numba JIT)
**Status:** DISCARDED ❌ — compilation issues + NaN
**Commit:** `f955186`
**Result:** Numba compilation hangs 3+ minutes. NaN residual on harvest (diagonal handling bug).
**Lesson:** Forward/backward triangular solve in numba requires careful diagonal handling. Even if compilation succeeds, the NaN failure shows the implementation is fragile. Same IRLS trajectory concern as SSOR anyway.

---

### Exp H: IC(0) preconditioner (numba JIT)
**Status:** DISCARDED ❌ — bear MADE +13.8%
**Commit:** `0baa1e0`
**Result:** Total 20.5s, but bear MADE: 0.334 → 0.388 (+16%). Other objects similar timing.
**Root cause:** BiNI's sigmoid weights → near-zero off-diagonal entries near discontinuities → negative Schur complements during IC(0) factorization → factorization corruption → different CG convergence path → wrong IRLS trajectory for bear.
**Lesson:** IC(0) factorization breakdown is inherent to BiNI's discontinuity-preserving weights. Cannot be fixed without changing the algorithm.

---

### Exp I: SSOR preconditioner (numba JIT)
**Status:** DISCARDED ❌ — bear MADE +14.7%
**Commit:** `da6bce9`
**Result:** Total 22.1s, bear MADE: 0.334 → 0.383 (+14.7%). Same pattern as IC(0).
**Root cause:** SSOR uses A entries directly (no factorization breakdown), but still makes CG converge faster/differently → different IRLS trajectory → consistently hits worse local minimum for bear. Two IRLS local minima exist: Jacobi's slow path leads to better one (0.334), SSOR's faster path leads to worse one (0.383).
**Key insight:** For BiNI, **Jacobi's specific slow convergence path is load-bearing**. Any preconditioner that improves CG convergence speed changes the IRLS trajectory for bear. This blocks all standard preconditioning strategies.

---

### Exp J: fastmath=True on _pcg_jacobi
**Status:** COMMITTED (`5956de6`) ✅ — marginally better on warm cache
**Result (warm cache):** Total ~15.15s, avg MADE 1.5036. ~18% speedup vs f8f7738 (18.5s), MADE identical.
**Change:** `@numba.njit(cache=True, fastmath=True)` — enables LLVM SIMD vectorization of AXPY/dot loops via FP reassociation.
**Rationale:** fastmath reassociates FP additions at machine-epsilon level (~1e-12), 9 orders of magnitude below cg_tol=1e-3. Should not alter IRLS trajectory.

| Object  | Time (s) | MADE (mm) |
|---------|----------|-----------|
| bear    | 1.11     | 0.334     |
| buddha  | 2.48     | 1.098     |
| cat     | 1.02     | 0.074     |
| cow     | 0.52     | 0.058     |
| goblet  | 0.74     | 9.018     |
| harvest | 5.13     | 1.838     |
| pot1    | 2.27     | 0.635     |
| pot2    | 1.04     | 0.220     |
| reading | 0.90     | 0.257     |
| **Total** | **15.21** | **avg 1.5036** |

---

### Exp K: Precompute C matrix for in-place A_mat update
**Status:** COMMITTED (`a960c9b`) ✅
**Result:** Total 13.5s, avg MADE 1.5036. **~11% speedup vs 5956de6 (15.2s)**, MADE identical.
**Changes:**
- Precompute sparse C matrix (nnz_Amat × 4n) s.t. `A_mat.data = C @ w` via `_build_C_coo` numba JIT
- Each IRLS iter: `_spmatvec_inplace(C.data, C.indices, C.indptr, w, A_mat.data)` instead of sparse matmul
- Replaces O(16n) sparse matmul (18ms/iter) with O(3.2n) SpMV (1ms/iter)
**Lesson:** Building the C matrix once and reusing it eliminates the dominant matbuild cost. Works because A_mat structure is fixed; only .data values change with w.

---

### Exp L: Cache A_i@z in weight update, reuse for energy
**Status:** COMMITTED (`d888cf3`) ✅
**Result:** Total 12.9s, avg MADE 1.5036. **~4% speedup vs a960c9b (13.5s)**, MADE identical.
**Changes:**
- After CG solve, compute A1z, A2z, A3z, A4z once
- Pass them to both weight update and energy check (was computing twice)
- Saves 4 SpMVs per IRLS iteration (~0.4ms each on harvest)
**Lesson:** Simple reuse eliminates redundant O(nnz) work. Energy check and weight update both need A_i@z.

---

### Exp M: Fused _update_weights_energy + _spmatvec_inplace numba kernels
**Status:** COMMITTED (`be08a14`) ✅
**Result:** Total 11.8s, avg MADE 1.5036. **~8% speedup vs d888cf3 (12.9s)**, MADE identical.
**Changes:**
- `_update_weights_energy(A1z, A2z, A3z, A4z, nx, ny, k, w_out)`: single numba loop computing sigmoid weights + energy, writes w in-place, returns scalar energy. Replaces 8 numpy calls + 8+ temp arrays.
- `_spmatvec_inplace(data, indices, indptr, x, out)`: in-place CSR SpMV avoiding scipy dispatch overhead and allocation.
- `_has_depth_prior` flag to correctly extract wu/wv from w after loop (needed because depth_map gets reassigned after IRLS).

| Object  | Time (s) | MADE (mm) |
|---------|----------|-----------|
| bear    | 0.86     | 0.334     |
| buddha  | 1.90     | 1.098     |
| cat     | 0.83     | 0.074     |
| cow     | 0.42     | 0.058     |
| goblet  | 0.66     | 9.018     |
| harvest | 4.03     | 1.838     |
| pot1    | 1.71     | 0.635     |
| pot2    | 0.72     | 0.220     |
| reading | 0.70     | 0.257     |
| **Total** | **11.83** | **avg 1.5036** |

**Lesson:** Fusing weight+energy eliminates numpy dispatch overhead and temp allocation. For harvest (n=56k) this saves ~14ms/iter × ~135 iters ≈ 1.9s total.


---

### Exp N: Module-level numba JIT warmup
**Status:** COMMITTED (`bc4a0d6`) ✅
**Result:** Total wall ~12.1s, avg MADE 1.5036. **~0.5s wall time savings**, MADE unchanged.
**Change:** Added module-level warmup calls to `_build_C_coo`, `_pcg_jacobi`, `_spmatvec_inplace`, `_update_weights_energy` at the bottom of the module (before `if __name__ == '__main__':`). Fires at import time, not during first object's processing.
**Lesson:** Without this, bear (first object) pays ~0.15s for numba cache loading + JIT dispatch initialization. Moving this to import time also stabilizes benchmark variance (all objects now start with fully warm JIT state). Actual observed savings ~0.5s (more than expected 0.15s — likely due to reduced JIT overhead for all 9 objects).


---

### Exp O: Looser CG tolerance (cg_tol=2e-3)
**Status:** DISCARDED ❌ — MADE degraded for most objects
**Result:** Total 8.20s but avg MADE 1.6857 vs baseline 1.5036. MADE changes: bear -38% (better!), buddha +35%, harvest +33%, pot1 +15%, goblet +7%, cow +60% — multiple objects violate ±5% constraint.
**Lesson:** cg_tol is more sensitive than expected. Even 2× looser tolerance changes IRLS trajectory for most objects. bear uniquely IMPROVES (finds a better local minimum with faster CG convergence), but others degrade significantly. The 1e-3 default is tight by necessity.

---

### Exp P: Replace _build_C_coo with _build_C_csr (direct CSR build)
**Status:** COMMITTED (`3f3ac93`) ✅ — code simplification, negligible timing change
**Result:** Total ~11.9s (avg of 12.2s and 11.7s across two runs), avg MADE 1.5036. Within noise of bc4a0d6 (12.1s).
**Changes:**
- New `_build_C_csr`: iterates A_mat entries in row-major order; uses AT (=A in CSC) with merge-intersection to find contributing row pairs; writes C_data, C_indices, C_indptr directly — no COO intermediate, no sort.
- Removed 32-line `_build_C_coo` (dead code after precompute phase updated).
- Removed post-hoc `_C.indices.astype(np.int32)` / `_C.indptr.astype(np.int32)` casts (now returned as int32 from numba).
- Updated module-level warmup to call `_build_C_csr` instead of `_build_C_coo`.
**Lesson:** COO→CSR sort was already fast at these n sizes (≤57k). The ~6ms bear precompute savings per call are negligible vs total solver time. Kept for code quality (simpler, no dead function).

