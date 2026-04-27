"""
ensemble_sindy.py - Ensemble SINDy (Fasel, Kutz, Brunton & Brunton, 2022).

Reference
---------
Fasel U., Kutz J. N., Brunton B. W., Brunton S. L. (2022).
Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise
limit, with active learning and control.
Proc. Roy. Soc. A 478:20210904.  DOI: 10.1098/rspa.2021.0904

Core idea
---------
For one (S0, lambda, phi) point we draw B bootstrap row-resamples of the
aligned (Theta, X_next) matrices, run the existing sparsify_dynamics on
each one, and aggregate:

    inclusion_prob[k, j] = (1/B) * #{ b : |Xi_b[k, j]| > eps }
    Xi_median[k, j]      = median_b( Xi_b[k, j] )
    Xi_q25, Xi_q75       = 25th/75th percentile across b

We then keep terms whose inclusion probability >= 0.6 in EITHER equation
and refit OLS on that filtered support to debias from the threshold-
shrinkage of sparsifyDynamics. The refit Xi is "Xi_filtered" -- the
consensus model.

This directly addresses three weaknesses the paper itself flags:

    1. Chickenpox overfit: spurious terms (e.g. beta(t)*I^2 = 114.8)
       only survive a fraction of the bootstraps, so their inclusion
       probability falls below 0.6 and they are filtered out -- without
       the ad-hoc PSD switch the paper resorts to.
    2. Noise sensitivity: bootstrap aggregation averages out
       noise-driven term selections.
    3. No uncertainty quantification: the IQR bands are credible-
       interval-like bounds on each surviving coefficient.

Two-pass runtime strategy
-------------------------
Naive nesting of B bootstraps inside the full (S0, lambda, phi) grid
is infeasible (~1M SINDy calls). grid_search_ensemble() runs the
existing baseline grid first to find best (S0*, lambda*, phi*), then
runs B-bootstrap E-SINDy in a small 3x3x3 neighborhood around that
point. The best-AIC ensemble model in that neighborhood is returned.
"""

from typing import Optional

import numpy as np

from src.config import ENSEMBLE, SINDY
from src.function_library import build_library
from src.preprocessing import preprocess_disease
from src.simulation import simulate_discovered_model
from src.sindy_core import sparsify_dynamics, compute_sparsity_index
from src.model_selection import compute_aic, grid_search


_EPS = 1e-10


def bootstrap_row_indices(n_samples, n_bootstrap, rng, block_size=1):
    """
    Generate (n_bootstrap, n_samples) array of row indices.

    block_size=1 -> classic i.i.d. row bagging (Fasel et al. default).
    block_size>1 -> block bootstrap of contiguous chunks (preserves some
                    short-horizon temporal structure).
    """
    if block_size <= 1:
        return rng.integers(low=0, high=n_samples, size=(n_bootstrap, n_samples))

    # Block bootstrap: pick start indices, glue blocks together.
    n_blocks = int(np.ceil(n_samples / block_size))
    out = np.empty((n_bootstrap, n_blocks * block_size), dtype=np.int64)
    for b in range(n_bootstrap):
        starts = rng.integers(low=0, high=n_samples - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        out[b] = idx
    return out[:, :n_samples]


def _refit_on_support(Theta, X_next, support_mask):
    """
    Ordinary least-squares refit restricted to columns where support_mask
    is True. Removes the bias introduced by hard thresholding inside
    sparsifyDynamics.
    """
    p = Theta.shape[1]
    n_eqs = X_next.shape[1] if X_next.ndim > 1 else 1
    Xi = np.zeros((p, n_eqs))
    keep = np.where(support_mask)[0]
    if len(keep) == 0:
        return Xi
    Theta_k = Theta[:, keep]
    Xi_k, _, _, _ = np.linalg.lstsq(Theta_k, X_next, rcond=None)
    Xi[keep, :] = Xi_k
    return Xi


def run_ensemble_sindy(
    Theta,
    X_next,
    lambda_c,
    n_bootstrap: int = 100,
    inclusion_threshold: float = 0.6,
    library_bagging: bool = False,
    library_subsample_frac: float = 0.8,
    block_size: int = 1,
    rng_seed: int = 42,
    sensitivity_thresholds: Optional[list] = None,
):
    """
    Run E-SINDy at a single (S0, lambda, phi) point.

    Parameters
    ----------
    Theta : (n, p) -- function library, already aligned with X_next.
    X_next : (n, n_eqs) -- target = next-step state, n_eqs = 2 (S, I).
    lambda_c : float -- sparsity threshold passed to sparsify_dynamics.
    n_bootstrap : int -- B in the paper.
    inclusion_threshold : float -- term must survive >= this fraction.
    library_bagging : bool -- also subsample Theta columns each draw.
    library_subsample_frac : float -- column-keep fraction when bagging.
    block_size : int -- 1 = i.i.d., >1 = block bootstrap.
    rng_seed : int -- reproducibility.
    sensitivity_thresholds : list of float -- extra thresholds to record
                                              n_surviving counts at.

    Returns
    -------
    dict with keys:
        Xi_bootstraps, Xi_median, Xi_q25, Xi_q75, inclusion_prob,
        Xi_filtered, active_filtered, n_bootstrap, inclusion_threshold,
        threshold_sensitivity
    """
    rng = np.random.default_rng(rng_seed)
    n, p = Theta.shape
    n_eqs = X_next.shape[1] if X_next.ndim > 1 else 1
    if X_next.ndim == 1:
        X_next = X_next.reshape(-1, 1)

    idx_matrix = bootstrap_row_indices(n, n_bootstrap, rng, block_size=block_size)
    Xi_bootstraps = np.zeros((n_bootstrap, p, n_eqs))

    for b in range(n_bootstrap):
        rows = idx_matrix[b]
        Theta_b = Theta[rows]
        X_b = X_next[rows]

        if library_bagging:
            n_keep = max(1, int(round(p * library_subsample_frac)))
            cols = rng.choice(p, size=n_keep, replace=False)
            Xi_b_partial, _ = sparsify_dynamics(Theta_b[:, cols], X_b, lambda_c)
            Xi_b = np.zeros((p, n_eqs))
            Xi_b[cols] = Xi_b_partial
        else:
            Xi_b, _ = sparsify_dynamics(Theta_b, X_b, lambda_c)

        Xi_bootstraps[b] = Xi_b

    # ---- Aggregate across bootstraps -----------------------------------
    inclusion_prob = (np.abs(Xi_bootstraps) > _EPS).mean(axis=0)  # (p, n_eqs)
    Xi_median = np.median(Xi_bootstraps, axis=0)
    Xi_q25 = np.quantile(Xi_bootstraps, 0.25, axis=0)
    Xi_q75 = np.quantile(Xi_bootstraps, 0.75, axis=0)

    # ---- Self-consistency assertions -----------------------------------
    assert np.all((inclusion_prob >= 0.0) & (inclusion_prob <= 1.0))
    # Note: median is not strictly between q25 and q75 only when bootstraps
    # are degenerate (all zero), in which case q25 == median == q75 == 0.
    assert np.all(Xi_q25 - 1e-12 <= Xi_median), "q25 > median"
    assert np.all(Xi_median <= Xi_q75 + 1e-12), "median > q75"

    # ---- Filter by inclusion probability + refit OLS -------------------
    # A term survives if it clears the threshold in EITHER equation.
    survives_any = (inclusion_prob >= inclusion_threshold).any(axis=1)
    Xi_filtered = _refit_on_support(Theta, X_next, survives_any)

    # ---- Sensitivity sweep at additional thresholds --------------------
    threshold_sensitivity = {}
    if sensitivity_thresholds:
        for thr in sensitivity_thresholds:
            mask = (inclusion_prob >= thr).any(axis=1)
            threshold_sensitivity[float(thr)] = int(mask.sum())

    return {
        "Xi_bootstraps": Xi_bootstraps,
        "Xi_median": Xi_median,
        "Xi_q25": Xi_q25,
        "Xi_q75": Xi_q75,
        "inclusion_prob": inclusion_prob,
        "Xi_filtered": Xi_filtered,
        "active_filtered": survives_any,
        "n_bootstrap": n_bootstrap,
        "inclusion_threshold": inclusion_threshold,
        "threshold_sensitivity": threshold_sensitivity,
    }


def _build_neighborhood(best, S0_full, lambda_full, phi_full, cfg):
    """
    Build a small 3x3x3 grid around best (S0, lambda, phi).

    Picked from the full ranges so we never propose a value outside the
    user's intended search space.
    """
    S0_c = best["S0"]
    lam_c = best["lambda_c"]
    phi_c = best["phi"]

    # S0 neighborhood: +/- radius_frac * S0_c
    S0_radius = cfg["neighborhood_S0_radius_frac"] * abs(S0_c)
    S0_neighbors = np.linspace(S0_c - S0_radius, S0_c + S0_radius,
                               cfg["neighborhood_S0_pts"])
    # Clip to the user-supplied range.
    S0_neighbors = np.clip(S0_neighbors, S0_full.min(), S0_full.max())

    # lambda neighborhood in log-space
    log_lam = np.log10(lam_c)
    decades = cfg["neighborhood_lambda_decades"]
    lam_neighbors = np.logspace(log_lam - decades, log_lam + decades,
                                cfg["neighborhood_lambda_pts"])
    lam_neighbors = np.clip(lam_neighbors, lambda_full.min(), lambda_full.max())

    # phi neighborhood in weeks (wraps modulo 52 for cyclical phase)
    phi_radius = cfg["neighborhood_phi_radius_weeks"]
    phi_neighbors = np.linspace(phi_c - phi_radius, phi_c + phi_radius,
                                cfg["neighborhood_phi_pts"])
    phi_neighbors = np.mod(phi_neighbors, 52.0)

    # Deduplicate while keeping order
    return (np.unique(S0_neighbors),
            np.unique(lam_neighbors),
            np.unique(phi_neighbors))


def grid_search_ensemble(
    cases,
    births,
    population,
    D_i,
    p,
    L,
    S0_range=None,
    lambda_range=None,
    phi_range=None,
    library_order=2,
    n_bootstrap=None,
    inclusion_threshold=None,
    library_bagging=None,
    rng_seed=None,
    verbose=True,
):
    """
    Two-pass best-then-ensemble grid search.

    Pass 1: existing baseline grid_search to locate (S0*, lambda*, phi*).
    Pass 2: E-SINDy with B bootstraps in a 3x3x3 neighborhood around that
            point; among those, return the model with the lowest AIC on
            the *Xi_filtered* (consensus) model.

    Returns
    -------
    best : dict (compatible with grid_search return + ensemble fields)
    pass1_best : dict (the baseline pass-1 best, kept for comparison)
    """
    cfg = ENSEMBLE
    if S0_range is None:
        S0_range = SINDY["S0_range"]
    if lambda_range is None:
        lambda_range = SINDY["lambda_range"]
    if phi_range is None:
        phi_range = SINDY["phi_range"]
    if n_bootstrap is None:
        n_bootstrap = cfg["n_bootstrap"]
    if inclusion_threshold is None:
        inclusion_threshold = cfg["inclusion_threshold"]
    if library_bagging is None:
        library_bagging = cfg["library_bagging"]
    if rng_seed is None:
        rng_seed = cfg["rng_seed"]

    # ----- Pass 1: baseline grid search to locate best point -------------
    if verbose:
        print(f"\n  [Ensemble] Pass 1: baseline grid search "
              f"({len(S0_range)} x {len(lambda_range)} x {len(phi_range)} pts)")
    pass1_best, _ = grid_search(
        cases, births, population, D_i, p, L,
        S0_range=S0_range,
        lambda_range=lambda_range,
        phi_range=phi_range,
        library_order=library_order,
        verbose=verbose,
    )
    if pass1_best is None:
        if verbose:
            print("  [Ensemble] Pass 1 returned no model. Aborting.")
        return None, None

    # ----- Pass 2: ensemble grid in a small neighborhood -----------------
    S0_n, lam_n, phi_n = _build_neighborhood(pass1_best,
                                             np.asarray(S0_range),
                                             np.asarray(lambda_range),
                                             np.asarray(phi_range),
                                             cfg)
    if verbose:
        print(f"  [Ensemble] Pass 2: bootstrap (B={n_bootstrap}) over "
              f"{len(S0_n)} x {len(lam_n)} x {len(phi_n)} neighborhood "
              f"around (S0={pass1_best['S0']:.4f}, "
              f"lambda={pass1_best['lambda_c']:.5f}, "
              f"phi={pass1_best['phi']:.1f})")

    best_aic = np.inf
    best_result = None
    sens_thresholds = cfg.get("sensitivity_thresholds", None)

    for S0 in S0_n:
        S_t, I_t, cases_smooth, alpha = preprocess_disease(
            cases, births, population, D_i, p, L, S0_fraction=float(S0)
        )
        t_weeks = np.arange(len(S_t), dtype=float)
        X_next_full = np.column_stack([S_t[1:], I_t[1:]])

        for lam in lam_n:
            for phi in phi_n:
                Theta_full, labels = build_library(
                    S_t, I_t, t_weeks, float(phi), library_order
                )
                Theta_trimmed = Theta_full[:-1, :]
                if len(Theta_trimmed) < 10:
                    continue

                ensemble = run_ensemble_sindy(
                    Theta_trimmed,
                    X_next_full,
                    float(lam),
                    n_bootstrap=n_bootstrap,
                    inclusion_threshold=inclusion_threshold,
                    library_bagging=library_bagging,
                    rng_seed=rng_seed,
                    sensitivity_thresholds=sens_thresholds,
                )

                Xi_filt = ensemble["Xi_filtered"]
                n_params = int(np.sum(np.any(np.abs(Xi_filt) > _EPS, axis=1)))
                if n_params == 0:
                    continue

                # Score the consensus model exactly like grid_search does.
                try:
                    S_sim, I_sim = simulate_discovered_model(
                        Xi_filt, S_t[0], I_t[0], t_weeks, float(phi),
                        library_order
                    )
                except Exception:
                    continue
                n_compare = min(len(I_t), len(I_sim))
                aic = compute_aic(I_t[:n_compare], I_sim[:n_compare], n_params)

                if aic < best_aic:
                    best_aic = aic
                    best_result = {
                        # Standard fields (grid_search-compatible)
                        "Xi": Xi_filt.copy(),
                        "labels": labels,
                        "S0": float(S0),
                        "lambda_c": float(lam),
                        "phi": float(phi),
                        "aic": float(aic),
                        "S_t": S_t,
                        "I_t": I_t,
                        "S_sim": S_sim[:n_compare],
                        "I_sim": I_sim[:n_compare],
                        "sparsity": compute_sparsity_index(Xi_filt, len(labels)),
                        "n_params": n_params,
                        "alpha": float(alpha),
                        # Ensemble extras
                        "Xi_median": ensemble["Xi_median"].copy(),
                        "Xi_q25": ensemble["Xi_q25"].copy(),
                        "Xi_q75": ensemble["Xi_q75"].copy(),
                        "Xi_filtered": Xi_filt.copy(),
                        "Xi_bootstraps": ensemble["Xi_bootstraps"].copy(),
                        "inclusion_prob": ensemble["inclusion_prob"].copy(),
                        "n_bootstrap": int(n_bootstrap),
                        "inclusion_threshold": float(inclusion_threshold),
                        "threshold_sensitivity": ensemble["threshold_sensitivity"],
                    }

    if verbose and best_result is not None:
        print(f"  [Ensemble] Best ensemble model: "
              f"S0={best_result['S0']:.4f}, "
              f"lambda={best_result['lambda_c']:.5f}, "
              f"phi={best_result['phi']:.1f}, "
              f"AIC={best_result['aic']:.1f}, "
              f"sparsity r={best_result['sparsity']:.2f}, "
              f"n_active={best_result['n_params']}")
    return best_result, pass1_best


def print_ensemble_model(result, equation_names=("S equation", "I equation")):
    """
    Pretty-print the ensemble result with median ± IQR + inclusion probs.

    Distinguishes:
        * BOLD       - term survives the inclusion threshold
        * dim/gray   - term is below threshold (filtered out)
    """
    Xi_med = result["Xi_median"]
    Xi_q25 = result["Xi_q25"]
    Xi_q75 = result["Xi_q75"]
    P = result["inclusion_prob"]
    labels = result["labels"]
    thr = result.get("inclusion_threshold", 0.6)

    print("\n" + "=" * 90)
    print(f"ENSEMBLE-SINDy MODEL  "
          f"(B={result['n_bootstrap']}, "
          f"inclusion_threshold={thr:.2f})")
    print("=" * 90)
    n_eqs = Xi_med.shape[1]
    header = f"{'Term':<14}"
    for name in equation_names[:n_eqs]:
        header += f"   {name + ' median (q25..q75)':>32}   {'P_incl':>8}"
    print(header)
    print("-" * 90)

    for i, label in enumerate(labels):
        survives = any(P[i, j] >= thr for j in range(n_eqs))
        row = f"{label:<14}"
        for j in range(n_eqs):
            med = Xi_med[i, j]
            q25 = Xi_q25[i, j]
            q75 = Xi_q75[i, j]
            p_incl = P[i, j]
            if abs(med) < _EPS and p_incl < 0.05:
                row += f"   {'0':>32}   {p_incl:>7.2f}"
            else:
                cell = f"{med:>+8.4f} ({q25:>+7.4f}..{q75:>+7.4f})"
                row += f"   {cell:>32}   {p_incl:>7.2f}"
        if survives:
            print(f"\033[1m{row}\033[0m")
        else:
            print(f"\033[90m{row}\033[0m")
    print("-" * 90)
    n_survived = int(result["active_filtered"].sum()) \
        if "active_filtered" in result \
        else int(np.sum(np.any(np.abs(result["Xi_filtered"]) > _EPS, axis=1)))
    print(f"Terms surviving threshold {thr:.2f}: {n_survived}/{len(labels)}")
    if result.get("threshold_sensitivity"):
        ts = result["threshold_sensitivity"]
        ts_str = ", ".join(f"{thr_v:.2f}->{n}" for thr_v, n in sorted(ts.items()))
        print(f"Sensitivity (threshold -> n_active): {ts_str}")
    print("=" * 90)
