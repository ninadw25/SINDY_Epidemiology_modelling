"""
model_selection.py — AIC computation and parameter grid search.

The paper searches over a 3D grid of:
    S₀ (initial susceptible fraction): 0.05 to 0.13
    λ  (sparsity threshold):           0.0001 to 0.1
    φ  (seasonal phase shift):         0 to 52 weeks

At each grid point, SINDy discovers a model, which is scored by AIC.
The grid point with the lowest AIC gives the best model.

AIC balances fit quality vs model complexity:
    AIC = 2k − 2·ln(L)
where k = number of free parameters, L = likelihood.
"""

import numpy as np
from src.preprocessing import preprocess_disease
from src.function_library import build_library
from src.sindy_core import sparsify_dynamics, compute_sparsity_index
from src.simulation import simulate_discovered_model
from src.config import SINDY


def compute_aic(y_true, y_pred, n_params):
    """
    Compute Akaike Information Criterion.
    
    Equation 10 from the paper:
        AIC = 2k − 2·ln(L)
    
    Under Gaussian likelihood assumption:
        AIC = 2k + n·ln(RSS/n)
    
    where RSS = residual sum of squares, n = number of data points, k = free params.
    
    Parameters
    ----------
    y_true : array — observed data
    y_pred : array — model predictions
    n_params : int — number of nonzero coefficients in the model (k)
    
    Returns
    -------
    aic : float — AIC score (lower = better)
    """
    n = len(y_true)
    residuals = y_true - y_pred
    rss = np.sum(residuals ** 2)
    
    if rss < 1e-20:
        rss = 1e-20  # prevent log(0)
    
    # AIC = 2k + n·ln(RSS/n)
    aic = 2 * n_params + n * np.log(rss / n)
    
    return aic


def time_series_split(n, n_folds=5, min_train_frac=0.5):
    """
    Forward-chained train/test indices for time-series cross-validation.

    KFold leaks the future into the past for non-stationary series, which
    is exactly what epidemic data is. Forward chaining always trains on
    the past and tests on the future:

        Fold 1: train [0, n*min_train_frac),         test [n*min_train_frac,           n*min_train_frac + step)
        Fold 2: train [0, n*min_train_frac + step),  test [n*min_train_frac + step,    n*min_train_frac + 2*step)
        ...

    Parameters
    ----------
    n : int -- length of the series.
    n_folds : int -- number of forward folds.
    min_train_frac : float in (0, 1) -- fraction of n used by the first
        fold's training window. The remaining (1 - min_train_frac) is
        split equally across the n_folds test windows.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
    """
    n_train_min = int(n * min_train_frac)
    if n_train_min < 5 or n - n_train_min < n_folds:
        # Series too short -- fall back to a single 80/20 split
        train_end = max(int(n * 0.8), 5)
        if train_end >= n:
            return
        yield (np.arange(train_end), np.arange(train_end, n))
        return

    test_size = max(1, (n - n_train_min) // n_folds)
    for k in range(n_folds):
        train_end = n_train_min + k * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if test_end <= test_start:
            return
        yield (np.arange(0, train_end), np.arange(test_start, test_end))


def _cv_aic_score(Theta_trimmed, X_next, S_t, I_t, t_weeks, phi,
                  library_order, lam, n_folds, min_train_frac):
    """
    Forward-chained CV-AIC using ONE-STEP-AHEAD prediction.

    Why one-step-ahead and not forward simulation?
    ----------------------------------------------
    SINDy's regression target is X(t+1) = Theta(t) @ Xi -- discrete-time,
    one step at a time. Free-running forward simulation amplifies any
    small Xi error exponentially, and on noisy epi data with very low
    prevalence values (~1e-5) the RSS is dominated by the runaway path
    rather than by model quality. A trivial Xi that produces I_sim ~= 0
    can win on RSS because data values are themselves tiny.

    One-step-ahead matches the regression objective exactly: predicted
    X_next vs actual X_next on each held-out timestep. This is also the
    standard CV target for autoregressive / state-space sparse-regression
    in the SINDy literature (Mangan et al. 2017, Brunton et al. 2016).

    Each fold:
        1. Fit Xi_fold = sparsify_dynamics(Theta_train, X_next_train).
        2. predicted_test = Theta_test @ Xi_fold.
        3. Compute AIC on column 1 (I-equation) residuals.
    Average AIC across folds.
    """
    aic_values = []
    n_total = len(Theta_trimmed)
    for train_idx, test_idx in time_series_split(n_total, n_folds=n_folds,
                                                 min_train_frac=min_train_frac):
        if len(train_idx) < 10 or len(test_idx) < 2:
            continue
        Xi_fold, _ = sparsify_dynamics(Theta_trimmed[train_idx],
                                       X_next[train_idx], lam)
        n_params_fold = int(np.sum(np.any(np.abs(Xi_fold) > 1e-10, axis=1)))
        if n_params_fold == 0:
            continue
        # One-step-ahead prediction on test rows
        pred_next = Theta_trimmed[test_idx] @ Xi_fold  # (n_test, 2)
        target = X_next[test_idx, 1]                    # I(t+1) on test rows
        pred = pred_next[:, 1]
        if len(target) < 2:
            continue
        aic_fold = compute_aic(target, pred, n_params_fold)
        if not np.isfinite(aic_fold):
            continue
        aic_values.append(aic_fold)
    return float(np.mean(aic_values)) if aic_values else float("nan")


def grid_search(cases, births, population, D_i, p, L,
                S0_range=None, lambda_range=None, phi_range=None,
                library_order=2, verbose=True,
                use_cv_aic=None, cv_folds=None, cv_min_train_frac=None):
    """
    Search over (S0, lambda, phi) grid to find the best SINDy model.

    For each (S0, lambda) pair:
        1. Preprocess data with S0
        2. For each phi: build library, run SINDy, score, keep best-phi
    Select the overall best (S0, lambda) by minimum AIC.

    AIC scoring mode is controlled by `use_cv_aic`:
        * True  -> forward-chained CV-AIC (default; matches plan).
                   Fits SINDy on each train fold, scores on the held-out
                   future fold, averages across folds. The reported Xi
                   is then re-fit on the FULL series for downstream use.
        * False -> legacy in-sample AIC (back-compat).

    Parameters
    ----------
    cases, births, population, D_i, p, L : disease inputs.
    S0_range, lambda_range, phi_range : grid axes. Default = SINDY config.
    library_order : int -- 2 or 3.
    verbose : bool.
    use_cv_aic : bool or None -- defaults to SINDY["use_cv_aic"].
    cv_folds : int or None -- defaults to SINDY["cv_folds"].
    cv_min_train_frac : float or None -- defaults to SINDY["cv_min_train_frac"].

    Returns
    -------
    best_result : dict (Xi, labels, S0, lambda_c, phi, aic, S_t, I_t,
                        S_sim, I_sim, sparsity, n_params, alpha)
    all_results : list of dicts (for plot_grid_search)
    """
    if S0_range is None:
        S0_range = SINDY["S0_range"]
    if lambda_range is None:
        lambda_range = SINDY["lambda_range"]
    if phi_range is None:
        phi_range = SINDY["phi_range"]
    if use_cv_aic is None:
        use_cv_aic = SINDY.get("use_cv_aic", True)
    if cv_folds is None:
        cv_folds = SINDY.get("cv_folds", 5)
    if cv_min_train_frac is None:
        cv_min_train_frac = SINDY.get("cv_min_train_frac", 0.5)

    best_aic = np.inf
    best_result = None
    all_results = []

    n_total = len(S0_range) * len(lambda_range)
    counter = 0

    for S0 in S0_range:
        for lam in lambda_range:
            counter += 1

            if verbose and counter % 10 == 0:
                print(f"  Grid search: {counter}/{n_total} "
                      f"(S0={S0:.3f}, lambda={lam:.5f})")

            # --- Preprocess with this S0 ---
            S_t, I_t, cases_smooth, alpha = preprocess_disease(
                cases, births, population, D_i, p, L, S0_fraction=S0
            )

            t_weeks = np.arange(len(S_t), dtype=float)

            best_phi_aic = np.inf
            best_phi_result = None

            for phi in phi_range:
                # --- Build library ---
                Theta, labels = build_library(S_t, I_t, t_weeks, phi, library_order)

                # --- Set up the regression target ---
                X_next = np.column_stack([S_t[1:], I_t[1:]])
                Theta_trimmed = Theta[:-1, :]

                if len(Theta_trimmed) < 10:
                    continue

                # --- Score this (S0, lambda, phi): CV or in-sample -----
                if use_cv_aic:
                    aic = _cv_aic_score(Theta_trimmed, X_next, S_t, I_t,
                                        t_weeks, phi, library_order, lam,
                                        cv_folds, cv_min_train_frac)
                    if not np.isfinite(aic):
                        continue
                    # The reported Xi must come from a refit on the full
                    # series, since CV only used it for scoring.
                    Xi, active = sparsify_dynamics(Theta_trimmed, X_next, lam)
                    n_params = int(np.sum(np.any(np.abs(Xi) > 1e-10, axis=1)))
                    if n_params == 0:
                        continue
                    try:
                        S_sim, I_sim = simulate_discovered_model(
                            Xi, S_t[0], I_t[0], t_weeks, phi, library_order
                        )
                    except Exception:
                        continue
                    n_compare = min(len(I_t), len(I_sim))
                else:
                    # Legacy in-sample AIC path (preserved for back-compat).
                    Xi, active = sparsify_dynamics(Theta_trimmed, X_next, lam)
                    try:
                        S_sim, I_sim = simulate_discovered_model(
                            Xi, S_t[0], I_t[0], t_weeks, phi, library_order
                        )
                    except Exception:
                        continue
                    n_params = int(np.sum(np.any(np.abs(Xi) > 1e-10, axis=1)))
                    if n_params == 0:
                        continue
                    n_compare = min(len(I_t), len(I_sim))
                    aic = compute_aic(I_t[:n_compare], I_sim[:n_compare],
                                      n_params)

                if aic < best_phi_aic:
                    best_phi_aic = aic
                    best_phi_result = {
                        "Xi": Xi.copy(),
                        "labels": labels,
                        "active": active.copy(),
                        "S0": S0,
                        "lambda_c": lam,
                        "phi": phi,
                        "aic": aic,
                        "S_t": S_t,
                        "I_t": I_t,
                        "S_sim": S_sim[:n_compare],
                        "I_sim": I_sim[:n_compare],
                        "sparsity": compute_sparsity_index(Xi, len(labels)),
                        "n_params": n_params,
                        "alpha": alpha,
                    }

            if best_phi_result is not None:
                all_results.append({
                    "S0": S0, "lambda_c": lam,
                    "aic": best_phi_result["aic"],
                    "sparsity": best_phi_result["sparsity"],
                })

                if best_phi_result["aic"] < best_aic:
                    best_aic = best_phi_result["aic"]
                    best_result = best_phi_result

    if verbose and best_result is not None:
        scoring = "CV-AIC" if use_cv_aic else "in-sample AIC"
        print(f"\n  Best model ({scoring}): S0={best_result['S0']:.4f}, "
              f"lambda={best_result['lambda_c']:.5f}, "
              f"phi={best_result['phi']:.1f} weeks, "
              f"AIC={best_result['aic']:.1f}, "
              f"sparsity r={best_result['sparsity']:.2f}")

    return best_result, all_results
