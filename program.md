# program.md — Bilateral Normal Integration AutoResearch

This is an autonomous optimization experiment for the BiNI solver. The agent modifies the solver, runs the DiLiGenT benchmark, and iterates indefinitely to find faster implementations without degrading accuracy.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `master`.
3. **Read these files for full context:**
   - `CLAUDE.md` — project overview, commands, architecture.
   - `bilateral_normal_integration_numpy.py` — the file you modify.
   - `evaluation_diligent.py` — the evaluation harness. Do **not** modify.
   - `optimization_log.md` — history of past attempts (successful and failed).
4. **Verify data exists**: Check that `data/Fig7_diligent/` contains 9 object folders each with `normal_map.png`, `mask.png`, `K.txt`, and that `diligent_depth_GT/` contains 9 `.mat` files. If missing, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**.

---

## Experimentation

Run the benchmark as: `conda run -n mvups python evaluation_diligent.py > run.log 2>&1`

**What you CAN do:**
- Modify `bilateral_normal_integration_numpy.py` — this is the only file you edit. Everything in the solver is fair game: linear algebra operations, sparse matrix strategy, preconditioner, solver choice, data structures, loop structure, etc.

**What you CANNOT do:**
- Modify `evaluation_diligent.py`. It is read-only. It defines the evaluation protocol.
- Use GPU acceleration (no CuPy, no CUDA). CPU-only.
- Install packages that require a GPU (e.g. cupy, torch with CUDA). Pure CPU packages are allowed — install with `pip install <package>` and add to `requirements.txt`.
- Change hyperparameters (`k`, `max_iter`, `tol`) — these are set by the evaluation script.

**The goal: minimize total wall time across all 9 objects while keeping MADE within ±5% of the baseline per object.**

The two metrics are in tension: a faster solver that diverges or takes more IRLS iterations may produce worse MADE. You must respect both.

**Simplicity criterion**: All else equal, simpler is better. A 5% speedup that adds 30 lines of fragile code is not worth it. Removing code and getting equal or better results is a win. Weigh complexity cost against improvement magnitude.

**The first run**: Always establish the baseline first — run the unmodified solver and record it.

---

## Output format

The script prints one line per object:

```
{obj_name} wall time: {elapsed:.2f} sec
{obj_name} MADE: {made:.3f}
```

And at the end, a dict of all MADE values. Extract results:

```bash
grep "wall time\|MADE:" run.log
```

Key numbers to record:
- **total_time_s** = sum of all 9 wall times
- **avg_made_mm** = mean of all 9 MADE values (from the final pprint dict)

---

## Logging results

Log each experiment to `results.tsv` (tab-separated). Do **not** commit this file.

Header and columns:

```
commit	total_time_s	avg_made_mm	status	description
```

1. git commit hash (short, 7 chars)
2. total wall time in seconds, rounded to `.1f`
3. average MADE across all 9 objects, rounded to `.4f`
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```
commit	total_time_s	avg_made_mm	status	description
1341ac9	76.7	1.5041	keep	baseline
45089f9	37.4	1.5041	keep	precompute AT + array ops for W
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar11`).

LOOP FOREVER:

1. Read `results.tsv` and `optimization_log.md` to understand what has been tried.
2. Propose one specific change to `bilateral_normal_integration_numpy.py`. Ideas to explore:
   - Preconditioner improvements (ILU, AMG via `pyamg`, SSOR, incomplete Cholesky via `scikit-sparse`/`sksparse`)
   - Direct solver (`spsolve`) for the first IRLS iteration (cold start)
   - Avoiding repeated sparse matrix assembly (reuse sparsity pattern)
   - Tighter CG tolerance schedule (coarse early, fine late)
   - Reducing iterations via better convergence criteria
   - Algebraic simplifications in the IRLS update
   - Vectorized weight update computation
   - Avoiding `A @ z - b` recomputation (reuse from CG residual)
3. Edit `bilateral_normal_integration_numpy.py`.
4. `git commit` with a descriptive message.
5. Run: `conda run -n mvups python evaluation_diligent.py > run.log 2>&1`
6. Check output: `grep "wall time\|MADE:" run.log`
7. If grep is empty or MADE values are missing — it crashed. Run `tail -n 50 run.log` for the traceback. Fix if trivial; otherwise skip, log `crash`, and `git reset --hard HEAD~1`.
8. Record in `results.tsv`.
9. **Keep** if: total_time_s improved AND per-object MADE did not increase by more than 5% vs baseline for any object. Advance the branch.
10. **Discard** if: slower, or MADE degraded beyond tolerance. `git reset --hard HEAD~1`.

**Crashes**: Fix trivial bugs (typos, missing imports) and re-run. If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the loop begins, do not pause to ask the human. Do not ask "should I keep going?". Run indefinitely until manually interrupted. If you run out of obvious ideas, re-read the solver theory in `README.md`, look at the CuPy version (`bilateral_normal_integration_cupy.py`) for algorithmic hints, research CPU sparse solver packages available on PyPI, try combining previous near-misses, or try more radical restructuring. The loop runs until the human interrupts you, period.

---

## Baseline (for reference)

Recorded on `1341ac9`:

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
| **Total** | **76.69** | avg **1.504** |
