"""Analytical computation of geometric metrics directly from boundary Fourier coefficients.

These functions avoid the expensive VMEC equilibrium solve and compute
aspect_ratio, average_triangularity, and max_elongation directly from
the SurfaceRZFourier boundary representation.

Only edge_rotational_transform_over_n_field_periods (iota/nfp) still
requires a surrogate or VMEC, because it depends on the magnetic
equilibrium solution.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

# ---------------------------------------------------------------------------
# We work with raw numpy arrays (r_cos, z_sin) so these functions can be
# called from the optimizer hot-path without constructing full pydantic
# objects.  A convenience wrapper that accepts SurfaceRZFourier is at the
# bottom of the file.
# ---------------------------------------------------------------------------

# Default resolution for angular grids
_N_THETA = 201
_N_PHI = 64


# =========================================================================
# Low-level helpers (pure numpy, no pydantic)
# =========================================================================

def _evaluate_rz(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate R(theta, phi) and Z(theta, phi) from Fourier coefficients.

    Parameters
    ----------
    r_cos, z_sin : (n_pol, n_tor) Fourier coefficient matrices.
    n_field_periods : NFP
    theta, phi : broadcastable angle arrays.

    Returns
    -------
    R, Z : arrays with shape = broadcast(theta, phi).
    """
    n_pol, n_tor = r_cos.shape
    max_n = (n_tor - 1) // 2

    # mode grids
    m = np.arange(n_pol)                          # (n_pol,)
    n = np.arange(-max_n, max_n + 1)              # (n_tor,)

    # angle = m*theta - NFP*n*phi   →  shape (*broadcast, n_pol, n_tor)
    angle = (
        m[np.newaxis, :, np.newaxis] * theta[..., np.newaxis, np.newaxis]
        - n_field_periods
        * n[np.newaxis, np.newaxis, :]
        * phi[..., np.newaxis, np.newaxis]
    )
    R = np.sum(r_cos * np.cos(angle), axis=(-2, -1))
    Z = np.sum(z_sin * np.sin(angle), axis=(-2, -1))
    return R, Z


# =========================================================================
# Aspect ratio
# =========================================================================

def aspect_ratio_from_coeffs(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = _N_THETA,
    n_phi: int = _N_PHI,
) -> float:
    """Compute geometric aspect ratio = R_major / r_minor from the boundary.

    R_major = (R_max + R_min) / 2  (over the full surface)
    r_minor = (R_max - R_min) / 2

    This is the standard geometric definition used by VMEC's ``aspect``.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")

    R, _ = _evaluate_rz(r_cos, z_sin, n_field_periods, theta_g, phi_g)

    R_max = np.max(R)
    R_min = np.min(R)
    R_major = (R_max + R_min) / 2.0
    r_minor = (R_max - R_min) / 2.0
    if r_minor < 1e-12:
        return 1e6  # degenerate
    return float(R_major / r_minor)


# =========================================================================
# Average triangularity
# =========================================================================

def average_triangularity_from_coeffs(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = _N_THETA,
) -> float:
    r"""Compute average triangularity at the two stellarator-symmetry planes.

    Mirrors the definition in ``geometry_utils.average_triangularity``:

    .. math::
        \delta = \frac{\delta_{top} + \delta_{bottom}}{2}

    where :math:`\delta_{top} = 2(R_0 - R_{Z_{max}}) / (R_{max} - R_{min})`
    and the average is taken over the phi=0 and phi=pi/nfp planes.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)
    phis = np.array([0.0, np.pi / n_field_periods])

    theta_g, phi_g = np.meshgrid(theta, phis, indexing="ij")
    R, Z = _evaluate_rz(r_cos, z_sin, n_field_periods, theta_g, phi_g)

    # R0 ≈ centroid of the cross-section (matches the library implementation)
    R0 = np.mean(R, axis=0)           # (2,)
    R_max = np.max(R, axis=0)
    R_min = np.min(R, axis=0)
    minor_r = (R_max - R_min) / 2.0

    idx_max_Z = np.argmax(Z, axis=0)  # (2,)
    R_at_Zmax = R[idx_max_Z, np.arange(2)]

    triangularity = (R0 - R_at_Zmax) / minor_r
    return float(np.mean(triangularity))


# =========================================================================
# Max elongation (boundary-only approximation)
# =========================================================================

def _polygon_area_2d(R: np.ndarray, Z: np.ndarray) -> float:
    """Signed area of a 2-D polygon via the shoelace formula."""
    return float(0.5 * np.abs(np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)))


def max_elongation_from_coeffs(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = _N_THETA,
    n_phi: int = _N_PHI,
) -> float:
    """Approximate max elongation from the boundary cross-sections.

    For each toroidal slice the cross-section (R, Z) curve is computed,
    an ellipse with matching perimeter and area is fitted, and the
    elongation (semi-major / semi-minor) is returned.

    Unlike the VMEC-based version this does NOT use the magnetic axis;
    instead each cross-section is centred on its centroid, which is a
    good approximation for the Phase-0 surrogate warm-up.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_range = np.linspace(
        0, 2 * np.pi / n_field_periods, n_phi, endpoint=False
    )

    max_elong = 1.0
    a_guess = 1.0

    for phi_val in phi_range:
        phi_arr = np.full_like(theta, phi_val)
        R, Z = _evaluate_rz(r_cos, z_sin, n_field_periods, theta, phi_arr)

        # Perimeter
        dR = np.diff(np.append(R, R[0]))
        dZ = np.diff(np.append(Z, Z[0]))
        perimeter = float(np.sum(np.sqrt(dR**2 + dZ**2)))

        # Area (shoelace)
        area = _polygon_area_2d(R, Z)
        if area < 1e-14:
            continue

        # Fit ellipse: area = pi*a*b,  perimeter ≈ 4*a*E(e)
        def _residual(a):
            b = area / (np.pi * a)
            if b > a:
                a, b = b, a
            e_sq = 1.0 - (b / a) ** 2
            return perimeter - 4.0 * a * special.ellipe(e_sq)

        try:
            a_sol = float(optimize.fsolve(_residual, a_guess, full_output=False)[0])
        except Exception:
            continue

        if a_sol <= 0:
            continue

        a_guess = a_sol
        b_sol = area / (np.pi * a_sol)
        semi_maj = max(a_sol, b_sol)
        semi_min = min(a_sol, b_sol)
        if semi_min < 1e-14:
            continue
        elong = semi_maj / semi_min
        if elong > max_elong:
            max_elong = elong

    return float(max_elong)


# =========================================================================
# Convenience: compute all three from raw coefficient arrays
# =========================================================================

def compute_analytical_metrics(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = _N_THETA,
    n_phi: int = 32,
) -> dict[str, float]:
    """Return ``{elongation, aspect_ratio, triangularity}`` for a boundary.

    Uses smaller default *n_phi* for speed (this is called ~30 000 times
    during the surrogate warm-up).
    """
    return {
        "elongation": max_elongation_from_coeffs(
            r_cos, z_sin, n_field_periods, n_theta=n_theta, n_phi=n_phi,
        ),
        "aspect_ratio": aspect_ratio_from_coeffs(
            r_cos, z_sin, n_field_periods, n_theta=n_theta, n_phi=n_phi,
        ),
        "triangularity": average_triangularity_from_coeffs(
            r_cos, z_sin, n_field_periods, n_theta=n_theta,
        ),
    }


# =========================================================================
# VMEC-like aspect ratio (volume-based minor radius)
# =========================================================================

def vmec_aspect_ratio_from_coeffs(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = _N_THETA,
    n_phi: int = _N_PHI,
) -> float:
    """Compute aspect ratio matching VMEC's definition more closely.

    VMEC defines:
        aspect = R_major / r_minor
    where r_minor = sqrt(V / (2 * pi^2 * R_major)) and V is the plasma
    volume.  For a boundary-only computation we approximate V by summing
    cross-sectional areas multiplied by the toroidal path length.

    This definition accounts for the *shape* of each cross-section,
    not just the R extent, giving much better agreement with VMEC than
    the simple (R_max-R_min)/2 formula.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # Use one full field period; multiply by nfp for full volume
    phi_range = np.linspace(
        0, 2 * np.pi / n_field_periods, n_phi, endpoint=False
    )
    dphi = 2 * np.pi / (n_field_periods * n_phi)

    total_volume = 0.0
    total_R0 = 0.0

    for phi_val in phi_range:
        phi_arr = np.full_like(theta, phi_val)
        R, Z = _evaluate_rz(r_cos, z_sin, n_field_periods, theta, phi_arr)

        # Cross-section area via shoelace formula
        area = _polygon_area_2d(R, Z)
        # Centroid R of this cross-section
        R_centroid = float(np.mean(R))

        # dV ≈ area * R_centroid * dphi (toroidal volume element)
        total_volume += area * R_centroid * dphi
        total_R0 += R_centroid

    # Multiply by nfp for full torus volume
    total_volume *= n_field_periods
    R_major = total_R0 / n_phi

    if R_major < 1e-12 or total_volume < 1e-20:
        return 1e6  # degenerate

    # r_minor from volume: V = 2 * pi^2 * R_major * r_minor^2
    r_minor = np.sqrt(total_volume / (2.0 * np.pi**2 * R_major))
    if r_minor < 1e-12:
        return 1e6

    return float(R_major / r_minor)


# =========================================================================
# Boundary validity checks
# =========================================================================

def boundary_is_valid(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = 101,
    n_phi: int = 16,
    min_R: float = 0.01,
    min_area: float = 1e-6,
) -> bool:
    """Check that a boundary is physically valid for VMEC.

    Rejects boundaries that:
      1. Have R <= 0 anywhere (plasma crosses the symmetry axis)
      2. Have cross-sections with near-zero area (degenerate / pinched)
      3. Have self-intersecting cross-sections (winding number != 1)

    This is a fast pre-filter to avoid sending non-physical shapes to VMEC,
    which would crash or produce garbage.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi_range = np.linspace(
        0, 2 * np.pi / n_field_periods, n_phi, endpoint=False
    )

    for phi_val in phi_range:
        phi_arr = np.full_like(theta, phi_val)
        R, Z = _evaluate_rz(r_cos, z_sin, n_field_periods, theta, phi_arr)

        # Check 1: R must be strictly positive
        if np.any(R < min_R):
            return False

        # Check 2: cross-section area must be non-degenerate
        area = _polygon_area_2d(R, Z)
        if area < min_area:
            return False

        # Check 3: self-intersection detection via winding number
        # A simple (non-self-intersecting) closed curve has winding
        # number == 1 about its centroid.  We check the total angle
        # swept around the centroid.
        R_c = np.mean(R)
        Z_c = np.mean(Z)
        dR = R - R_c
        dZ = Z - Z_c
        angles = np.arctan2(dZ, dR)
        dangle = np.diff(angles)
        # Wrap to [-pi, pi]
        dangle = (dangle + np.pi) % (2 * np.pi) - np.pi
        winding = abs(np.sum(dangle)) / (2 * np.pi)
        # Should be ~1.0 for a simple curve.  Allow tolerance for
        # discretisation.
        if abs(winding - 1.0) > 0.15:
            return False

        # Check 4: no sharp self-crossing (adjacent segments intersecting)
        # Detect large direction reversals in the tangent vector
        dR_seg = np.diff(np.append(R, R[0]))
        dZ_seg = np.diff(np.append(Z, Z[0]))
        # Dot product of consecutive tangent vectors
        dot = dR_seg[:-1] * dR_seg[1:] + dZ_seg[:-1] * dZ_seg[1:]
        cross = dR_seg[:-1] * dZ_seg[1:] - dZ_seg[:-1] * dR_seg[1:]
        seg_len = np.sqrt(dR_seg[:-1]**2 + dZ_seg[:-1]**2) * np.sqrt(
            dR_seg[1:]**2 + dZ_seg[1:]**2
        )
        # Normalised dot product (cosine of turn angle)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_angle = np.where(seg_len > 1e-14, dot / seg_len, 1.0)
        # A sharp reversal (cos < -0.8 ≈ >143°) signals a kink / fold
        if np.any(cos_angle < -0.8):
            return False

    return True


# =========================================================================
# Fast vectorised version for the hot-loop
# =========================================================================

def compute_analytical_metrics_fast(
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    n_field_periods: int,
    n_theta: int = 101,
    n_phi: int = 16,
) -> dict[str, float]:
    """Same as *compute_analytical_metrics* but with reduced resolution
    for maximum throughput in the Phase-0 surrogate loop.

    Now also includes the VMEC-like volume-based aspect ratio.

    Accuracy is ~1-2 % relative to the full-resolution version, which is
    more than sufficient for ranking candidates.
    """
    metrics = compute_analytical_metrics(
        r_cos, z_sin, n_field_periods, n_theta=n_theta, n_phi=n_phi,
    )
    metrics["vmec_aspect_ratio"] = vmec_aspect_ratio_from_coeffs(
        r_cos, z_sin, n_field_periods, n_theta=n_theta, n_phi=n_phi,
    )
    return metrics
