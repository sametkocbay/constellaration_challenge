"""Two-phase optimizer for the GeometricalProblem.

Phase 0  — Surrogate warm-up:
  Nevergrad + dual surrogate (iota/nfp, max_elongation) + analytical metrics
  find a VMEC-convergent starting point.

  Key filters applied to every candidate:
  * boundary_is_valid()         — rejects self-intersecting / degenerate shapes
  * elong > ELONG_VMEC_MAX      — rejects shapes too elongated for VMEC
    (prevents the "Minimum of B is at the boundary" error)
  * iota_unc > IOTA_UNC_MAX     — rejects out-of-distribution iota predictions

Phase 1  — VMEC + Augmented-Lagrangian Method (ALM), sequential:
  Uses the library's objective_constraints() sequentially.
  The library handles VMEC errors gracefully and returns NAN_TO_HIGH on failure.

Constraints (GeometricalProblem):
  aspect_ratio          <= 4.0
  average_triangularity <= -0.5
  edge_iota / nfp       >= 0.3

Objective: minimize max_elongation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import nevergrad as ng
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must happen before any constellaration imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in [str(SRC_ROOT), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import constellaration.forward_model as forward_model
import constellaration.initial_guess as init_guess
import constellaration.optimization.augmented_lagrangian as al
import constellaration.problems as problems
from constellaration.geometry import surface_rz_fourier as rz_fourier
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.optimization.augmented_lagrangian_runner import (
    objective_constraints as _lib_objective_constraints,
)
from constellaration.utils import pytree, seed_util

from challenge.surrogates import analytical_metrics, geoSurrogate

# Try to load GP surrogate; fall back to NN ensemble if not trained yet
try:
    from challenge.surrogates import geoSurrogateGP
    _GP_AVAILABLE = True
except ImportError:
    _GP_AVAILABLE = False

try:
    pytree.register_pydantic_data(
        rz_fourier.SurfaceRZFourier,
        meta_fields=["n_field_periods", "is_stellarator_symmetric"],
    )
except ValueError:
    pass

# =========================================================================
# CONFIGURATION
# =========================================================================

# --- Boundary parameterization ---
NFP = 3        # number of field periods
MAX_M = 3      # max poloidal mode  →  80+ free parameters  (more DOF = lower elongation)
MAX_N = 3      # max toroidal mode
ALPHA = 1.5    # infinity-norm spectrum scaling exponent

# --- Initial guess  (confirmed VMEC-convergent: aspect=3, elong=0.5, iota=0.4) ---
INIT_ASPECT = 3.0
INIT_ELONG = 0.5
INIT_IOTA = 0.4

# --- Safe fallback initial guess ---
SAFE_ASPECT = 3.0
SAFE_ELONG = 0.5
SAFE_IOTA = 0.4

# --- Phase 0 surrogate warm-up ---
SURROGATE_BUDGET = 1_200
SURROGATE_BOUNDS = 0.4         # ±0.4 around x0 in scaled space — wide enough for 72-dim exploration
SURROGATE_PENALTY = 300.0

# Constraint margins (Phase 0 aims tighter than actual limits)
ASPECT_MARGIN = 0.3            # target aspect  ≤ 3.7
TRI_MARGIN = 0.05              # target tri      ≤ -0.55
IOTA_MARGIN = 0.08             # target iota/nfp ≥ 0.38 (low_fidelity gap is ~0.07, small buffer)

# Hard VMEC-convergence filter: analytical elongation above this → reject
ELONG_VMEC_MAX = 2.8

# Ensemble uncertainty filter: iota std above this → reject (out-of-distribution)
IOTA_UNC_MAX = 0.15

# Number of top surrogate candidates to validate with VMEC
TOP_K = 20

# --- Phase 1 ALM ---
ALM_MAXIT = 20
BUDGET_INITIAL = 400   # start large (wide exploration), shrink as ALM converges
BUDGET_DECREMENT = 20  # reduce per iteration
BUDGET_MIN = 80        # floor
BOUNDS_INITIAL = 0.3
PENALTY_INITIAL = 100.0

AL_SETTINGS = al.AugmentedLagrangianSettings(
    constraint_violation_tolerance_reduction_factor=0.8,
    penalty_parameters_increase_factor=5,
    bounds_reduction_factor=0.9,
    penalty_parameters_max=1e8,
    bounds_min=0.05,
)

# VMEC settings — multi-fidelity progression:
#   very_low_fidelity  for early ALM iters (fast exploration)
#   low_fidelity       for mid ALM iters   (reliable convergence)
#   medium_fidelity    for final ALM iters  (survives official eval)
VMEC_SETTINGS = forward_model.ConstellarationSettings(
    qi_settings=None,
    turbulent_settings=None,
    vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
        fidelity="low_fidelity"
    ),
)

# Multi-fidelity thresholds (iteration index)
FIDELITY_SWITCH_TO_LOW = 5       # iters 0-4: very_low,  5+: low
FIDELITY_SWITCH_TO_MEDIUM = 17   # iters 17+: medium (last 3 of 20)


def _vmec_settings_for_iter(k: int) -> forward_model.ConstellarationSettings:
    """Return VMEC settings with fidelity appropriate for ALM iteration k."""
    if k < FIDELITY_SWITCH_TO_LOW:
        fidelity = "very_low_fidelity"
    elif k >= FIDELITY_SWITCH_TO_MEDIUM:
        fidelity = "medium_fidelity"
    else:
        fidelity = "low_fidelity"
    return forward_model.ConstellarationSettings(
        qi_settings=None,
        turbulent_settings=None,
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity=fidelity
        ),
    )

FEAS_TOL = 0.0     # only accept strictly feasible (all con[i] <= 0, feas_c == 0.0)

# Dynamic iota safety margin: starts high to push deep into the feasible zone,
# decays across ALM iterations to allow the optimizer to squeeze out lower elongation.
IOTA_SAFETY_MARGIN_START = 0.05   # aggressive in early iters
IOTA_SAFETY_MARGIN_END   = 0.01   # relaxed in final iters


def _iota_safety_margin(k: int, max_k: int) -> float:
    """Linearly decay iota safety margin from START to END over ALM iterations."""
    frac = min(k / max(max_k - 1, 1), 1.0)
    return IOTA_SAFETY_MARGIN_START + (IOTA_SAFETY_MARGIN_END - IOTA_SAFETY_MARGIN_START) * frac

# Only trigger early stop once we have a truly good feasible solution
EARLY_STOP_OBJ_TARGET = 2.0

OUT_PATH = PROJECT_ROOT / "boundary_m4_optimized.json"


def _save_boundary(boundary, label: str, feas: float, obj: float) -> None:
    """Write boundary to disk immediately in competition submission format."""
    with open(OUT_PATH, "w") as _fh:
        _fh.write(boundary.model_dump_json())
    print(
        f"  *** SAVED [{label}]  obj={obj:.4f}  feas={feas:.6f}  → '{OUT_PATH.name}' ***",
        flush=True,
    )


# =========================================================================
# PHASE 0: SURROGATE EVALUATION
# =========================================================================

def evaluate_surrogate(x, scale, unravel_fn, model, scales, problem):
    """Evaluate a candidate cheaply using surrogate + analytical geometry.

    Returns (loss, elong, feas, metrics_dict) or None if the candidate
    should be rejected (non-physical geometry, too elongated, or
    out-of-distribution for the surrogate).
    """
    try:
        boundary = unravel_fn(jnp.asarray(x * scale))
        r_cos = np.asarray(boundary.r_cos)
        z_sin = np.asarray(boundary.z_sin)
        nfp = boundary.n_field_periods

        # 1. Fast validity check — reject self-intersecting / degenerate shapes
        if not analytical_metrics.boundary_is_valid(r_cos, z_sin, nfp):
            return None

        # 2. Analytical geometric metrics (low-res, fast)
        geo = analytical_metrics.compute_analytical_metrics_fast(r_cos, z_sin, nfp)
        elong = geo["elongation"]
        aspect_simple = geo["aspect_ratio"]
        aspect_vmec = geo["vmec_aspect_ratio"]
        tri = geo["triangularity"]

        if not np.isfinite([elong, aspect_vmec, tri]).all():
            return None

        # 3. Hard elongation filter — VMEC diverges above this
        if elong > ELONG_VMEC_MAX:
            return None

        # 4. Surrogate prediction: iota/nfp (col 0) and optionally elongation (col 1)
        # Zero-pad raw features to match training data dimensionality
        x_raw_flat = np.concatenate([r_cos.flatten(), z_sin.flatten()]).astype(np.float32)
        n_raw = int(scales.get("n_raw_features", len(x_raw_flat)))
        if len(x_raw_flat) < n_raw:
            x_raw_flat = np.concatenate(
                [x_raw_flat, np.zeros(n_raw - len(x_raw_flat), dtype=np.float32)]
            )
        x_raw = x_raw_flat.reshape(1, -1)
        # Use whichever surrogate was loaded (GP or NN)
        if isinstance(model, dict) and "gp_models" in model:
            y_mean, y_std = geoSurrogateGP.predict_with_uncertainty(model, scales, x_raw)
        else:
            y_mean, y_std = geoSurrogate.predict_with_uncertainty(model, scales, x_raw)
        iota = float(y_mean[0, 0])
        iota_unc = float(y_std[0, 0])

        if not np.isfinite(iota):
            return None

        # 5. Reject high-uncertainty iota predictions (out-of-distribution)
        if iota_unc > IOTA_UNC_MAX:
            return None

        # If surrogate predicts elongation too (2-output model), add as soft penalty.
        # We do NOT use it as a hard filter here because the surrogate elongation
        # prediction can be inaccurate for zero-padded (low MAX_M/N) boundaries.
        # The analytical elongation hard filter in step 3 is the reliable gate.
        surr_elong_viol = 0.0
        if y_mean.shape[1] >= 2:
            elong_pred = float(y_mean[0, 1])
            if np.isfinite(elong_pred) and elong_pred > ELONG_VMEC_MAX:
                surr_elong_viol = (elong_pred - ELONG_VMEC_MAX) * 0.5

        # 6. Constraint violations (tighter than actual limits for Phase 0)
        asp_viol = max(aspect_vmec - (problem._aspect_ratio_upper_bound - ASPECT_MARGIN), 0.0)
        tri_viol = max(tri - (problem._average_triangularity_upper_bound - TRI_MARGIN), 0.0)
        iota_viol = max(
            (problem._edge_rotational_transform_over_n_field_periods_lower_bound + IOTA_MARGIN)
            - iota,
            0.0,
        )
        asp_s_viol = max(1.5 - aspect_simple, 0.0)   # must have aspect_simple > 1.5

        feas = float(
            np.sqrt(asp_viol ** 2 + tri_viol ** 2 + iota_viol ** 2 + asp_s_viol ** 2)
        )
        loss = (
            SURROGATE_PENALTY * (asp_viol ** 2 + tri_viol ** 2 + iota_viol ** 2 + asp_s_viol ** 2)
            + 0.1 * elong          # soft objective: prefer low elongation
            + 10.0 * iota_unc      # prefer confident iota predictions
            + surr_elong_viol      # soft penalty from surrogate elongation over-prediction
        )

        return loss, elong, feas, {
            "aspect_simple": aspect_simple,
            "aspect_vmec": aspect_vmec,
            "tri": tri,
            "iota": iota,
            "iota_unc": iota_unc,
        }
    except Exception:
        return None


# =========================================================================
# VMEC HELPER (sequential, used for Phase-0 validation)
# =========================================================================

def _vmec_eval_sequential(x_np, scale, problem, unravel_fn):
    """Run one VMEC evaluation sequentially.  Returns (obj, con_list, metrics)."""
    (obj, con), met = _lib_objective_constraints(
        jnp.array(x_np), scale, problem, unravel_fn, VMEC_SETTINGS, None
    )
    return float(obj), [float(c) for c in con], met


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Two-phase stellarator optimizer")
    _parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            f"Skip Phase 0 and start Phase 1 from the best boundary saved so far "
            f"({OUT_PATH.name}). File must exist."
        ),
    )
    _args = _parser.parse_args()

    seed_util.seed_everything(48)
    t_start = time.time()

    problem = problems.GeometricalProblem()

    # --- Choose surrogate: GP (preferred) or NN ensemble (fallback) ---
    USE_GP_SURROGATE = False
    if _GP_AVAILABLE:
        try:
            surr_model, surr_scales = geoSurrogateGP.load_surrogate()
            USE_GP_SURROGATE = True
            print("Using GP surrogate (calibrated uncertainty)")
        except FileNotFoundError:
            print("GP surrogate not trained yet — falling back to NN ensemble")
            surr_model, surr_scales = geoSurrogate.load_surrogate()
    else:
        surr_model, surr_scales = geoSurrogate.load_surrogate()

    # ------------------------------------------------------------------
    # Build initial boundary  (aspect=3, elong=0.5, iota=0.4  → VMEC-safe)
    # ------------------------------------------------------------------
    raw_boundary = init_guess.generate_rotating_ellipse(
        aspect_ratio=INIT_ASPECT,
        elongation=INIT_ELONG,
        rotational_transform=INIT_IOTA,
        n_field_periods=NFP,
    )
    boundary = rz_fourier.set_max_mode_numbers(
        surface=raw_boundary, max_poloidal_mode=MAX_M, max_toroidal_mode=MAX_N
    )
    mask = rz_fourier.build_mask(boundary, max_poloidal_mode=MAX_M, max_toroidal_mode=MAX_N)
    x_init_np, unravel_fn = pytree.mask_and_ravel(pytree=boundary, mask=mask)

    scale_raw = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=ALPHA,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale_raw[mask.r_cos], scale_raw[mask.z_sin]]))

    x0 = jnp.array(x_init_np) / scale
    n_params = int(len(x0))
    print(f"Parameters: {n_params}  (MAX_M={MAX_M}, MAX_N={MAX_N}, NFP={NFP})", flush=True)

    # ==================================================================
    # RESUME: skip Phase 0, load best boundary from disk
    # ==================================================================
    if _args.resume:
        if not OUT_PATH.exists():
            raise FileNotFoundError(
                f"--resume requested but no saved boundary found at {OUT_PATH}"
            )
        print(f"\n{'='*72}", flush=True)
        print(f"  RESUME: loading boundary from '{OUT_PATH.name}'", flush=True)
        print(f"{'='*72}", flush=True)
        _resume_boundary = rz_fourier.SurfaceRZFourier.model_validate_json(
            OUT_PATH.read_text()
        )
        # project onto the same mask/scale used by this run
        _resume_x_np, _ = pytree.mask_and_ravel(pytree=_resume_boundary, mask=mask)
        _resume_x_scaled = np.array(jnp.array(_resume_x_np) / scale)
        print("  Running VMEC to evaluate loaded boundary ...", flush=True)
        _r_obj, _r_con, _r_met = _vmec_eval_sequential(
            _resume_x_scaled, scale, problem, unravel_fn
        )
        if _r_met is None:
            raise RuntimeError(
                "Loaded boundary failed VMEC — cannot resume. "
                "Remove --resume and start fresh, or fix the boundary."
            )
        _r_feas = float(np.linalg.norm(np.maximum(0.0, _r_con)))
        print(
            f"  Loaded boundary: obj={_r_obj:.4f}  feas={_r_feas:.6f}  "
            f"asp={_r_met.aspect_ratio:.3f}  "
            f"tri={_r_met.average_triangularity:.3f}  "
            f"iota={_r_met.edge_rotational_transform_over_n_field_periods:.3f}",
            flush=True,
        )
        vmec_x0  = _resume_x_scaled
        vmec_obj0 = jnp.array(_r_obj)
        vmec_con0 = jnp.array(_r_con)
        # pre-populate the global best trackers so saving works immediately
        best_feas_boundary_resume = _resume_boundary
        best_feas_feas_resume     = _r_feas
        best_feas_obj_resume      = _r_obj

    # ==================================================================
    # PHASE 0: Surrogate warm-up  (skipped when --resume)
    # ==================================================================
    if not _args.resume:
        print(f"\n{'='*72}", flush=True)
        print(f"  PHASE 0: Surrogate warm-up  ({SURROGATE_BUDGET} evals, bounds=\xb1{SURROGATE_BOUNDS})", flush=True)
        print(f"  Elongation hard filter: elong \u2264 {ELONG_VMEC_MAX}", flush=True)
        print(f"  IoTA uncertainty filter: unc \u2264 {IOTA_UNC_MAX}", flush=True)
        print(f"{'='*72}", flush=True)

        surr_param = ng.p.Array(init=np.array(x0))
        surr_param.set_bounds(np.array(x0) - SURROGATE_BOUNDS, np.array(x0) + SURROGATE_BOUNDS)
        surr_opt = ng.optimizers.NGOpt(parametrization=surr_param, budget=SURROGATE_BUDGET)

        best_feas_s = float("inf")
        best_x_s = np.array(x0).copy()
        best_elong_s = float("inf")
        best_metrics_s: dict = {}
        surr_evals = 0
        top_k: list[tuple[float, float, np.ndarray]] = []  # (feas, elong, x)

        for i in range(SURROGATE_BUDGET):
            cand = surr_opt.ask()
            result = evaluate_surrogate(
                cand.value, scale, unravel_fn, surr_model, surr_scales, problem
            )
            if result is None:
                surr_opt.tell(cand, 1e6)
                continue

            loss, elong, feas, mdict = result
            surr_opt.tell(cand, loss)
            surr_evals += 1

            if feas < best_feas_s or (feas == best_feas_s and elong < best_elong_s):
                best_feas_s = feas
                best_x_s = cand.value.copy()
                best_elong_s = elong
                best_metrics_s = mdict

            entry = (feas, elong, cand.value.copy())
            if len(top_k) < TOP_K:
                top_k.append(entry)
                top_k.sort(key=lambda e: (e[0], e[1]))
            elif (feas, elong) < (top_k[-1][0], top_k[-1][1]):
                top_k[-1] = entry
                top_k.sort(key=lambda e: (e[0], e[1]))

            if (i + 1) % 100 == 0:
                print(
                    f"  [{i+1:6d}/{SURROGATE_BUDGET}]  valid={surr_evals}  feas={best_feas_s:.4f}  "
                    f"elong={best_elong_s:.4f} | "
                    f"asp_s={best_metrics_s.get('aspect_simple', 0):.3f}  "
                    f"asp_v={best_metrics_s.get('aspect_vmec', 0):.3f}  "
                    f"tri={best_metrics_s.get('tri', 0):.3f}  "
                    f"iota={best_metrics_s.get('iota', 0):.3f}  "
                    f"unc={best_metrics_s.get('iota_unc', 0):.3f}",
                    flush=True,
                )

        rec = surr_opt.provide_recommendation()
        res = evaluate_surrogate(rec.value, scale, unravel_fn, surr_model, surr_scales, problem)
        if res is not None:
            _, r_el, r_f, r_m = res
            if r_f < best_feas_s or (r_f == best_feas_s and r_el < best_elong_s):
                best_feas_s = r_f
                best_x_s = rec.value.copy()
                best_elong_s = r_el
                best_metrics_s = r_m

        t_p0 = time.time() - t_start
        print(f"\n  Phase 0 done in {t_p0:.1f}s  ({surr_evals} valid evals)")
        print(f"  Best surrogate: feas={best_feas_s:.4f}  elong={best_elong_s:.4f}")
        print(f"  Metrics: {best_metrics_s}")

        # ------------------------------------------------------------------
        # Validate top-K surrogate picks with VMEC (sequential)
        # ------------------------------------------------------------------
        validation_xs = [best_x_s] + [
            e[2] for e in top_k if not np.array_equal(e[2], best_x_s)
        ]
        print(f"\n  Validating top {len(validation_xs)} candidates with VMEC ...")

        vmec_x0 = None
        vmec_obj0 = None
        vmec_con0 = None
        n_vmec_ok = 0
        n_geom_rej = 0

        for idx, trial_x in enumerate(validation_xs):
            print(f"  [{idx+1:2d}/{len(validation_xs)}]", end="  ")

            # High-resolution geometric pre-check before expensive VMEC call
            try:
                b_t = unravel_fn(jnp.asarray(trial_x * scale))
                r_t, z_t, nfp_t = np.asarray(b_t.r_cos), np.asarray(b_t.z_sin), b_t.n_field_periods
                if not analytical_metrics.boundary_is_valid(r_t, z_t, nfp_t, n_theta=201, n_phi=32):
                    print("REJECTED (non-physical geometry)")
                    n_geom_rej += 1
                    continue
                asp_hi = analytical_metrics.vmec_aspect_ratio_from_coeffs(
                    r_t, z_t, nfp_t, n_theta=201, n_phi=32
                )
                if asp_hi > problem._aspect_ratio_upper_bound + 1.5:
                    print(f"REJECTED (asp_vmec={asp_hi:.2f} >> limit)")
                    n_geom_rej += 1
                    continue
            except Exception as e:
                print(f"REJECTED (pre-check error: {e})")
                n_geom_rej += 1
                continue

            obj_v, con_v, met_v = _vmec_eval_sequential(trial_x, scale, problem, unravel_fn)
            if met_v is not None:
                feas_v = float(np.linalg.norm(np.maximum(0.0, con_v)))
                print(
                    f"OK  obj={obj_v:.4f}  feas={feas_v:.4f} | "
                    f"asp={met_v.aspect_ratio:.3f}  "
                    f"tri={met_v.average_triangularity:.3f}  "
                    f"iota={met_v.edge_rotational_transform_over_n_field_periods:.3f}"
                )
                n_vmec_ok += 1
                if vmec_x0 is None or feas_v < float(np.linalg.norm(np.maximum(0.0, vmec_con0))):
                    vmec_x0 = trial_x.copy()
                    vmec_obj0 = jnp.array(obj_v)
                    vmec_con0 = jnp.array(con_v)
            else:
                print("FAILED (VMEC did not converge)")

        print(
            f"\n  Validation: {n_vmec_ok} OK, "
            f"{len(validation_xs) - n_vmec_ok - n_geom_rej} VMEC-failed, "
            f"{n_geom_rej} geom-rejected"
        )

        if vmec_x0 is None:
            print("  All candidates failed VMEC → using safe baseline boundary.")
            safe_b = init_guess.generate_rotating_ellipse(
                aspect_ratio=SAFE_ASPECT,
                elongation=SAFE_ELONG,
                rotational_transform=SAFE_IOTA,
                n_field_periods=NFP,
            )
            safe_b = rz_fourier.set_max_mode_numbers(
                surface=safe_b, max_poloidal_mode=MAX_M, max_toroidal_mode=MAX_N
            )
            x_safe_np, _ = pytree.mask_and_ravel(pytree=safe_b, mask=mask)
            vmec_x0 = np.array(jnp.array(x_safe_np) / scale)
            obj_fb, con_fb, met_fb = _vmec_eval_sequential(vmec_x0, scale, problem, unravel_fn)
            vmec_obj0 = jnp.array(obj_fb)
            vmec_con0 = jnp.array(con_fb)
            f_fb = float(np.linalg.norm(np.maximum(0.0, con_fb)))
            print(f"  Safe baseline VMEC: obj={obj_fb:.4f}  feas={f_fb:.4f}")
        else:
            f0 = float(np.linalg.norm(np.maximum(0.0, np.array(vmec_con0))))
            print(f"  Phase 1 starting point: obj={float(vmec_obj0):.4f}  feas={f0:.4f}")

    # ==================================================================
    # PHASE 1: Augmented Lagrangian (sequential VMEC)
    # ==================================================================
    print(f"\n{'='*72}", flush=True)
    print(f"  PHASE 1: ALM  ({ALM_MAXIT} iters, budget {BUDGET_INITIAL}→{BUDGET_MIN}, sequential)", flush=True)
    print(f"{'='*72}\n", flush=True)

    state = al.AugmentedLagrangianState(
        x=jnp.array(vmec_x0),
        multipliers=jnp.zeros(3),
        penalty_parameters=jnp.ones(3) * PENALTY_INITIAL,
        objective=vmec_obj0,
        constraints=vmec_con0,
        bounds=jnp.ones(n_params) * BOUNDS_INITIAL,
    )

    best_feas_boundary = None
    best_feas_obj = float("inf")
    best_feas_feas = float("inf")
    best_any_boundary = None
    best_any_feas = float("inf")
    best_any_obj = float("inf")
    vmec_total = 0
    vmec_fails = 0
    budget = BUDGET_INITIAL

    # Pre-populate best trackers when resuming from a saved boundary
    if _args.resume:
        if best_feas_feas_resume <= 1e-9:
            best_feas_boundary = best_feas_boundary_resume
            best_feas_obj      = best_feas_obj_resume
            best_feas_feas     = best_feas_feas_resume
            print(f"  [resume] pre-loaded FEASIBLE best: obj={best_feas_obj:.4f}  feas={best_feas_feas:.6f}", flush=True)
        else:
            best_any_boundary = best_feas_boundary_resume
            best_any_feas     = best_feas_feas_resume
            best_any_obj      = best_feas_obj_resume
            print(f"  [resume] pre-loaded BEST-SO-FAR: obj={best_any_obj:.4f}  feas={best_any_feas:.6f}", flush=True)

    for k in range(ALM_MAXIT):
        t_iter = time.time()

        # --- Dynamic VMEC fidelity for this iteration ---
        iter_vmec_settings = _vmec_settings_for_iter(k)
        fidelity_label = iter_vmec_settings.vmec_preset_settings.fidelity

        # --- Dynamic iota safety margin ---
        iota_margin_k = _iota_safety_margin(k, ALM_MAXIT)

        parametrization = ng.p.Array(
            init=np.array(state.x),
            lower=np.array(state.x - state.bounds),
            upper=np.array(state.x + state.bounds),
        )
        # Use Cobyla (local derivative-free) for sample-efficient ALM subproblems
        # instead of NGOpt (global meta-algorithm) which wastes budget exploring far away
        oracle = ng.optimizers.Cobyla(
            parametrization=parametrization, budget=budget, num_workers=1
        )
        oracle.suggest(np.array(state.x))

        # track best within this iteration only — save once at end of iter
        iter_best_feas = float("inf")
        iter_best_obj  = float("inf")
        iter_best_x    = None

        for _ in range(budget):
            cand = oracle.ask()
            (obj_j, con_j), met = _lib_objective_constraints(
                jnp.array(cand.value),
                scale, problem, unravel_fn, iter_vmec_settings, None,
            )
            vmec_total += 1
            if met is None:
                vmec_fails += 1
                oracle.tell(cand, 100.0)
                continue

            # Tighten iota constraint internally by dynamic margin so the oracle
            # targets iota/nfp >= 0.30 + margin. Margin decays from 0.05 → 0.01
            # across iterations. Real con_j is used for feasibility tracking.
            con_j_strict = con_j.at[2].add(iota_margin_k)
            loss_v = float(al.augmented_lagrangian_function(obj_j, con_j_strict, state))
            oracle.tell(cand, loss_v)

            feas_c = float(jnp.linalg.norm(jnp.maximum(0.0, con_j)))
            obj_c  = float(obj_j)

            # update global trackers (no save yet)
            if feas_c <= 1e-9 and obj_c < best_feas_obj:
                best_feas_obj      = obj_c
                best_feas_feas     = feas_c
                best_feas_boundary = unravel_fn(jnp.asarray(cand.value * scale))

            if feas_c < best_any_feas or (
                abs(feas_c - best_any_feas) < 1e-8 and obj_c < best_any_obj
            ):
                best_any_feas = feas_c
                best_any_obj  = obj_c
                best_any_boundary = unravel_fn(jnp.asarray(cand.value * scale))

            # track best of THIS iteration
            if feas_c < iter_best_feas or (
                abs(feas_c - iter_best_feas) < 1e-8 and obj_c < iter_best_obj
            ):
                iter_best_feas = feas_c
                iter_best_obj  = obj_c
                iter_best_x    = cand.value.copy()

        # ── save once per iteration if we improved the global best ──────
        if best_feas_boundary is not None:
            _save_boundary(best_feas_boundary, "FEASIBLE", best_feas_feas, best_feas_obj)
        elif iter_best_x is not None and iter_best_feas < best_any_feas + 1e-8:
            _save_boundary(best_any_boundary, "BEST-SO-FAR", best_any_feas, best_any_obj)

        # Evaluate recommendation, update ALM state
        (obj_r, con_r), met_r = _lib_objective_constraints(
            jnp.array(oracle.provide_recommendation().value),
            scale, problem, unravel_fn, iter_vmec_settings, None,
        )
        feas_r = float(jnp.linalg.norm(jnp.maximum(0.0, con_r)))
        state = al.update_augmented_lagrangian_state(
            x=jnp.array(oracle.provide_recommendation().value),
            objective=obj_r,
            constraints=con_r,
            state=state,
            settings=AL_SETTINGS,
        )
        budget = int(max(BUDGET_MIN, budget - BUDGET_DECREMENT))
        dt = time.time() - t_iter

        con_str = ", ".join(f"{float(c):.3f}" for c in con_r)
        met_str = ""
        if met_r is not None:
            met_str = (
                f"  asp={met_r.aspect_ratio:.2f}"
                f"  tri={met_r.average_triangularity:.3f}"
                f"  iota={met_r.edge_rotational_transform_over_n_field_periods:.3f}"
            )
        bf_str = f"{best_feas_obj:.4f}" if best_feas_boundary is not None else "N/A"
        print(
            f"[{k+1:2d}/{ALM_MAXIT}]  obj={float(obj_r):.3f}  feas={feas_r:.4f}  "
            f"fid={fidelity_label}  iota_m={iota_margin_k:.3f}  "
            f"pen={float(state.penalty_parameters[0]):.0f}  "
            f"bnd={float(state.bounds[0]):.3f} | "
            f"vmec={vmec_total}  fail={vmec_fails} | "
            f"con=[{con_str}]{met_str} | "
            f"best_feas={bf_str}  best_any={best_any_feas:.4f} | "
            f"{dt:.0f}s"
        )

        if best_feas_boundary is not None and best_feas_obj < EARLY_STOP_OBJ_TARGET:
            print(f"  [early stop — feasible obj={best_feas_obj:.4f} < target {EARLY_STOP_OBJ_TARGET}]")
            break

    # ==================================================================
    # FINALIZE
    # ==================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  Optimization complete in {total_time:.1f}s")
    print(f"  Total VMEC: {vmec_total} calls, {vmec_fails} failures")
    print(f"{'='*72}")

    if best_feas_boundary is not None:
        print("\n  → Using best FEASIBLE solution.")
        final = best_feas_boundary
        _save_boundary(final, "FINAL-FEASIBLE", best_feas_feas, best_feas_obj)
    elif best_any_boundary is not None:
        print(f"\n  → No fully feasible solution found (best feas={best_any_feas:.4f}).")
        final = best_any_boundary
        _save_boundary(final, "FINAL-BEST-SO-FAR", best_any_feas, best_any_obj)
    else:
        print("\n  → No successful VMEC evaluation. Saving last ALM state.")
        final = unravel_fn(jnp.asarray(state.x * scale))
        _save_boundary(final, "FINAL-FALLBACK", float("inf"), float("inf"))

    # ------------------------------------------------------------------
    # Final VMEC verification (low_fidelity for accuracy)
    # ------------------------------------------------------------------
    try:
        m_fin, _ = forward_model.forward_model(
            final,
            settings=forward_model.ConstellarationSettings(
                qi_settings=None,
                turbulent_settings=None,
                vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
                    fidelity="low_fidelity"
                ),
            ),
        )
        fv = problem.compute_feasibility(m_fin)
        is_f = problem.is_feasible(m_fin)
        print(f"\n  Final VMEC verification (low_fidelity):")
        print(f"    max_elongation   = {m_fin.max_elongation:.6f}")
        print(f"    aspect_ratio     = {m_fin.aspect_ratio:.6f}   (limit ≤ 4.0)")
        print(f"    triangularity    = {m_fin.average_triangularity:.6f}   (limit ≤ -0.5)")
        iota_f = m_fin.edge_rotational_transform_over_n_field_periods
        print(f"    iota/nfp         = {iota_f:.6f}   (limit ≥ 0.3)")
        print(f"    feasibility      = {fv:.6f}")
        print(f"    is_feasible      = {is_f}")
        if is_f:
            score = problem.evaluate(final)
            print(f"    SCORE            = {score}")
    except Exception as e:
        print(f"\n  Final verification failed: {e}")
