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
}

# =============================================================================
# Output Configuration
# =============================================================================

OUTPUT_DIR = "outputs"
FIGURE_DPI = 150
