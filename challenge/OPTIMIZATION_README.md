# Stellarator Boundary Optimizer — How It Works

## What We're Solving

The **GeometricalProblem** asks: find a stellarator boundary (described by
Fourier coefficients) that minimises `max_elongation` subject to:

| Constraint | Limit |
|---|---|
| `aspect_ratio` | ≤ 4.0 |
| `average_triangularity` | ≤ −0.5 |
| `iota / nfp` | ≥ 0.3 |

The boundary is parameterised as a `SurfaceRZFourier` with `nfp=3`,
`MAX_M=3`, `MAX_N=3` → **48 free parameters**.

---

## Two-Phase Optimizer

### Phase 0 — Surrogate Warm-up (~10 s, 1000 evals)

Nevergrad (NGOpt) explores the parameter space **without VMEC**, using:

1. **`boundary_is_valid()`** — rejects self-intersecting or degenerate shapes
2. **Analytical elongation filter** — rejects `elong > 2.8`
   (prevents the VMEC "Minimum of B at boundary" crash)
3. **Neural surrogate** — predicts `iota/nfp` and `max_elongation`
   from Fourier coefficients (3-model ensemble, trained on HuggingFace dataset)
4. **Uncertainty filter** — rejects `iota_unc > 0.15`
   (out-of-distribution shapes where the surrogate cannot be trusted)
5. **Tighter constraint targets** — Phase 0 aims for
   `iota ≥ 0.45` (actual limit 0.30), `tri ≤ −0.55` (actual −0.50),
   giving a safety margin to survive the fidelity upgrade in Phase 1

The top-20 surrogate candidates are then **validated with one VMEC call each**
to pick the best VMEC-convergent starting point.

### Phase 1 — Augmented Lagrangian Method (ALM, ~30–60 min)

Sequential VMEC (`low_fidelity`) calls inside an NGOpt inner loop:

- **ALM** adds a penalty to the objective for constraint violations,
  then updates Lagrange multipliers each outer iteration
- Budget starts at 400 VMEC calls/iter, shrinks by 20 each iter (floor: 80)
  → wide exploration early, cheap refinement late
- **Early stop** when a strictly feasible solution (`feas == 0.0`) is found
  with `obj < 1.8 × max_elongation` (empirically good quality threshold)
- **Intermediate saving** — every ALM iteration writes the best boundary to
  `boundary_m4_optimized.json` so results survive crashes

---

## Key Files

| File | Purpose |
|---|---|
| `challenge/surrogates/optimizer.py` | Main optimizer (Phase 0 + Phase 1) |
| `challenge/surrogates/geoSurrogate.py` | Neural surrogate: train + predict |
| `challenge/surrogates/analytical_metrics.py` | Fast geometry without VMEC |
| `challenge/models/stellarator_surrogate*.pth` | Trained ensemble weights |
| `evaluate_result.py` | Evaluate saved boundary at multiple fidelities |
| `visualize_boundary.py` | 3-D surface plot of saved boundary |
| `tests/surrogates/` | 64 fast unit tests + 17 VMEC integration tests |

---

## Running

```bash
# Fresh run (Phase 0 + Phase 1 from scratch, ~45 min)
challenge/.venv/bin/python -u challenge/surrogates/optimizer.py

# Resume from best saved boundary (skip Phase 0)
challenge/.venv/bin/python -u challenge/surrogates/optimizer.py --resume

# Evaluate saved boundary
challenge/.venv/bin/python evaluate_result.py

# Visualise (saves boundary_3d.png)
challenge/.venv/bin/python visualize_boundary.py

# Fast tests (no VMEC, ~7 s)
cd tests && challenge/../.venv/bin/pytest surrogates/ -v -m "not slow"

# Full tests including VMEC (~5 min)
cd tests && challenge/../.venv/bin/pytest surrogates/ -v
```

---

## Current Best Result

```
max_elongation  = 2.995  ← objective (lower is better)
aspect_ratio    = 3.765  ✓  (≤ 4.0)
triangularity   = −0.513 ✓  (≤ −0.5)
iota/nfp        = 0.301  ✓  (≥ 0.3)
feasibility     = 0.000  FEASIBLE ✓
score           = 0.780
```

---

## How to Improve the Score

The score is `1 − (elongation − 1) / (elongation_max − 1)` — lower
elongation = higher score.  Current elongation ≈ 2.995, score ≈ 0.780.

### 1. Run longer with `--resume` (easiest, +0.05–0.10 score)

The optimizer stopped at iter 3 of 20.  Simply resume:

```bash
challenge/.venv/bin/python -u challenge/surrogates/optimizer.py --resume
```

Iters 4–20 will refine around the feasible solution.  Expected
elongation reduction: 0.1–0.3 per few iters once constraints are tight.

### 2. Increase mode numbers (MAX_M=4, MAX_N=4, +0.05–0.15 score)

More Fourier modes give the optimizer more freedom to shape the boundary:

```python
# optimizer.py
MAX_M = 4
MAX_N = 4   # → 80 free parameters
```

Cost: ~2× more expensive per VMEC call (more Fourier modes).
The surrogate may need retraining (already trained on 5×9 = MAX_M=4, MAX_N=4 data).

### 3. Parallel runs with different seeds (best use of the cluster)

Each run is independent — launch 8 in parallel, keep the best:

```bash
for seed in 0 7 42 99 123 200 314 500; do
    PYTHONHASHSEED=$seed \
    challenge/.venv/bin/python -u challenge/surrogates/optimizer.py \
        2>&1 | tee logs/run_seed${seed}.log &
done
```

Expected gain: best of 8 runs likely 0.05–0.15 better than single run.

### 4. Tighten constraint margins after feasibility (automatic)

Once feasible, the ALM multipliers push constraints tighter.
The `iota=0.301` is dangerously close to the 0.300 limit — later iters
should push it to ≥ 0.32 for robustness across fidelity levels.

### 5. Retrain surrogate with more epochs / larger model

Current surrogate trains in ~5 min on CPU.  Better iota prediction
→ Phase 0 finds a lower-elongation starting point for Phase 1:

```bash
challenge/.venv/bin/python challenge/surrogates/geoSurrogate.py
```

Change `EPOCHS=500` → `EPOCHS=1000` and `hidden_dims=[256, 128, 64]`.

### 6. Switch to `medium_fidelity` VMEC for final polish

After Phase 1 finds a good feasible solution, run a final refinement
with higher-fidelity VMEC to get a boundary that survives the official
evaluator's `high_fidelity` solve:

```python
VMEC_SETTINGS = ... fidelity="medium_fidelity" ...
```

Cost: ~3× per VMEC call; run for just 5–10 ALM iters.

### Priority Order

| Approach | Effort | Expected gain |
|---|---|---|
| `--resume` (run longer) | trivial | +0.05–0.10 |
| Parallel cluster runs (8×) | low | +0.05–0.15 |
| MAX_M=4, MAX_N=4 | low | +0.05–0.15 |
| Surrogate retraining | medium | +0.02–0.05 |
| Medium-fidelity polish | medium | robustness |