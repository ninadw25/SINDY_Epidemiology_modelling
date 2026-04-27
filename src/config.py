"""
config.py — All parameters and constants used across the pipeline.

These values come directly from the paper (Horrocks & Bauch, 2020).
"""

import numpy as np

# =============================================================================
# Disease Parameters
# =============================================================================

DISEASES = {
    "measles": {
        "location": "England & Wales",
        "years": (1948, 1967),
        "weeks": 52 * 19,           # ~19 years of weekly data
        "D_i": 2.0,                 # Duration of infectiousness (weeks)
        "p": 0.957,                 # Lifetime infection probability
        "L": 65.0 * 52,             # Mean lifespan in weeks (65 years)
        "attractor": "biennial",    # Expected 2-year cycle
        "population": 50_000_000,   # Approximate UK population
        "birth_rate_weekly": 0.03 / 52,  # Per capita birth rate per week (ν)
        "death_rate_weekly": 0.03 / 52,  # Per capita death rate per week (μ)
    },
    "chickenpox": {
        "location": "Ontario, Canada",
        "years": (1946, 1967),
        "weeks": 52 * 21,
        "D_i": 2.0,
        "p": 0.957,
        "L": 65.0 * 52,
        "attractor": "annual",
        "population": 5_000_000,
        "birth_rate_weekly": 0.03 / 52,
        "death_rate_weekly": 0.03 / 52,
    },
    "rubella": {
        "location": "Ontario, Canada",
        "years": (1946, 1960),
        "weeks": 52 * 14,
        "D_i": 2.0,
        "p": 0.957,
        "L": 65.0 * 52,
        "attractor": "multiennial",
        "population": 5_000_000,
        "birth_rate_weekly": 0.03 / 52,
        "death_rate_weekly": 0.03 / 52,
    },
}

# =============================================================================
# SIR Model Parameters (for synthetic data generation)
# =============================================================================

SIR_PARAMS = {
    "measles": {
        "beta0": 8.0,          # Mean transmission rate (per week)
        "beta1": 0.25,         # Seasonal amplitude
        "gamma": 0.1,          # Recovery rate (per week) → 10 day recovery
        "phi": 0.0,            # Phase shift (weeks)
    },
    "chickenpox": {
        "beta0": 5.0,
        "beta1": 0.15,
        "gamma": 0.1,
        "phi": 0.0,
    },
    "rubella": {
        "beta0": 4.0,
        "beta1": 0.10,
        "gamma": 0.1,
        "phi": 0.0,
    },
}

# =============================================================================
# Preprocessing Parameters
# =============================================================================

SMOOTHING = {
    "window_length": 19,       # Savitzky-Golay window (must be odd)
    "polyorder": 3,            # Savitzky-Golay polynomial order
}

# =============================================================================
# SINDy Algorithm Parameters
# =============================================================================

SINDY = {
    "library_order": 2,        # Polynomial order (2 or 3)

    # Grid search ranges
    "S0_range": np.linspace(0.05, 0.13, 20),     # Initial susceptible fraction
    "lambda_range": np.logspace(-4, -1, 20),      # Sparsity threshold
    "phi_range": np.arange(0, 52, 2.0),           # Phase shift (weeks)

    # Convergence
    "max_iterations": 50,      # Max prune-refit cycles

    # Forward-chained time-series cross-validation (replaces in-sample AIC).
    # When use_cv_aic=True, AIC is averaged across cv_folds forward folds.
    "use_cv_aic": True,
    "cv_folds": 5,
    "cv_min_train_frac": 0.5,  # first fold trains on >= 50% of the series
}

# =============================================================================
# Ensemble-SINDy (Fasel et al., 2022, Proc. Roy. Soc. A 478:20210904)
# =============================================================================
# Bootstrap aggregation around the existing sparsifyDynamics, with per-term
# inclusion probabilities and IQR coefficient bands. Used by:
#   src/ensemble_sindy.py  (run_ensemble_sindy, grid_search_ensemble)
#   run_comparison.py      (top-level orchestrator)

ENSEMBLE = {
    "n_bootstrap": 100,                # B in the Fasel et al. paper
    "fast_n_bootstrap": 20,            # used when --fast flag is set
    "inclusion_threshold": 0.6,        # term must survive >= 60% of bootstraps
    "library_bagging": False,          # also subsample library columns each draw
    "library_subsample_frac": 0.8,     # only used when library_bagging=True
    "block_size": 1,                   # 1 = i.i.d. row bagging (Fasel default)
    "rng_seed": 42,                    # reproducibility
    # Pass-2 neighborhood around best (S0, lambda, phi) from baseline grid
    "neighborhood_S0_pts": 3,
    "neighborhood_lambda_pts": 3,
    "neighborhood_phi_pts": 3,
    "neighborhood_S0_radius_frac": 0.15,    # +/- 15% around best S0
    "neighborhood_lambda_decades": 0.3,     # +/- 0.3 decades around best lambda
    "neighborhood_phi_radius_weeks": 4.0,   # +/- 4 weeks around best phi
    # Sensitivity: report inclusion-survival counts at these thresholds too
    "sensitivity_thresholds": [0.5, 0.6, 0.7, 0.8],
}

# =============================================================================
# Regime-shift / out-of-sample test (paper Figs. 8-9)
# =============================================================================
# Reproduces the paper's vaccination-era extrapolation: reduce the S-equation
# constant + S coefficients (proxies for reduced susceptible recharge), then
# simulate forward and compare power spectra to the baseline model.

REGIME_SHIFT = {
    # Multipliers applied to the S-equation "1" and "S" library columns.
    # f=1.0 = baseline (pre-vaccine); f<1.0 = vaccination-era reduction.
    # Paper measles example: 0.606 -> 0.317 corresponds to f ~= 0.523.
    "perturbation_factors": [1.0, 0.75, 0.523, 0.25],
    "forward_steps": 520,             # 10 years x 52 weeks
    "expected_periods_years": {       # used by assert_regime_shift_metrics
        "measles": 2.0,
        "chickenpox": 1.0,
        "rubella": 5.5,
    },
    "period_tolerance_years": 0.3,    # baseline period must match within +/-
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_DIR = "outputs"
FIGURE_DPI = 150
