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


def grid_search(cases, births, population, D_i, p, L,
                S0_range=None, lambda_range=None, phi_range=None,
                library_order=2, verbose=True):
    """
    Search over (S₀, λ, φ) grid to find the best SINDy model.
    
    For each (S₀, λ) pair:
        1. Preprocess data with S₀
        2. For each φ: build library, run SINDy, simulate, compute AIC
        3. Keep the best-φ model for this (S₀, λ)
    
    Select the overall best (S₀, λ) by minimum AIC.
    
    Parameters
    ----------
    cases : array — raw weekly case counts
    births : array — weekly birth counts
    population : int — total population
    D_i, p, L : float — disease parameters
    S0_range : array — grid of S₀ values to try
    lambda_range : array — grid of λ values to try
    phi_range : array — grid of φ values to try
    library_order : int — polynomial order (2 or 3)
    verbose : bool — print progress
    
    Returns
    -------
    best_result : dict containing:
        - Xi: discovered coefficient matrix
        - labels: term names
        - S0: best initial susceptible fraction
        - lambda_c: best sparsity threshold
        - phi: best phase shift
        - aic: AIC score
        - S_t, I_t: preprocessed time series
        - sparsity: sparsity index
    all_results : list of all evaluated models (for plotting the grid)
    """
    if S0_range is None:
        S0_range = SINDY["S0_range"]
    if lambda_range is None:
        lambda_range = SINDY["lambda_range"]
    if phi_range is None:
        phi_range = SINDY["phi_range"]
    
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
                      f"(S₀={S0:.3f}, λ={lam:.5f})")
            
            # --- Preprocess with this S₀ ---
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
                # X_next = [S(t+1), I(t+1)] for discrete-time model
                X_next = np.column_stack([S_t[1:], I_t[1:]])
                Theta_trimmed = Theta[:-1, :]  # align: Θ(t) → X(t+1)
                
                if len(Theta_trimmed) < 10:
                    continue
                
                # --- Run SINDy ---
                Xi, active = sparsify_dynamics(Theta_trimmed, X_next, lam)
                
                # --- Simulate discovered model ---
                try:
                    S_sim, I_sim = simulate_discovered_model(
                        Xi, S_t[0], I_t[0], t_weeks, phi, library_order
                    )
                except Exception:
                    continue
                
                # --- Compute AIC ---
                n_params = int(np.sum(np.any(np.abs(Xi) > 1e-10, axis=1)))
                if n_params == 0:
                    continue
                
                # AIC based on I equation fit (we have data for I, not S)
                n_compare = min(len(I_t), len(I_sim))
                aic = compute_aic(I_t[:n_compare], I_sim[:n_compare], n_params)
                
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
        print(f"\n  ✓ Best model: S₀={best_result['S0']:.4f}, "
              f"λ={best_result['lambda_c']:.5f}, "
              f"φ={best_result['phi']:.1f} weeks, "
              f"AIC={best_result['aic']:.1f}, "
              f"sparsity r={best_result['sparsity']:.2f}")
    
    return best_result, all_results
