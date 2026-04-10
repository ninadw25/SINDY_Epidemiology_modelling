"""
function_library.py — Build the candidate function library Θ(X).

This is the "menu of ingredients" that SINDy picks from.
Equivalent to MATLAB's poolData() function.

For 2nd-order polynomial library with seasonal forcing:
    Θ(X) = [1, S, I, S², I², SI, β(t), β(t)S, β(t)I, β(t)S², β(t)SI, β(t)I²]

Each column of Θ is one candidate term evaluated at every timestep.
If you have 1000 weeks of data, Θ is a 1000 × 12 matrix.
"""

import numpy as np


def seasonal_forcing(t_weeks, phi, T=52.0):
    """
    Compute the seasonal forcing function β(t).
    
    Equation 22 from the paper:
        β(t) = 1 + cos(2πt/T − φ)
    
    Note: We use β₀=1, β₁=1 here because the actual magnitude of β₀ and β₁
    will be absorbed into the SINDy coefficients. What matters is the SHAPE
    (cosine oscillation) not the absolute scale.
    
    Parameters
    ----------
    t_weeks : array — time vector in weeks
    phi : float — phase shift in weeks (when seasonal peak occurs)
    T : float — period of oscillation (52 weeks = 1 year)
    
    Returns
    -------
    beta_t : array — seasonal forcing evaluated at each timestep
    """
    return 1.0 + np.cos(2.0 * np.pi * t_weeks / T - phi)


def build_library(S_t, I_t, t_weeks, phi, order=2):
    """
    Build the function library matrix Θ(X).
    
    This is equivalent to MATLAB's poolData() plus the seasonal extension.
    
    For order=2, the library is:
        Constant terms:  [1, S, I, S², I², SI]
        Seasonal terms:  [β(t), β(t)S, β(t)I, β(t)S², β(t)SI, β(t)I²]
    
    For order=3, additionally includes:
        Constant:  [S³, S²I, SI², I³]
        Seasonal:  [β(t)S³, β(t)S²I, β(t)SI², β(t)I³]
    
    Parameters
    ----------
    S_t : array of shape (n,) — susceptible proportion at each timestep
    I_t : array of shape (n,) — infectious prevalence at each timestep
    t_weeks : array of shape (n,) — time in weeks
    phi : float — seasonal phase shift
    order : int — polynomial order (2 or 3)
    
    Returns
    -------
    Theta : array of shape (n, p) — the function library matrix
    labels : list of str — human-readable name for each column
    """
    n = len(S_t)
    beta_t = seasonal_forcing(t_weeks, phi)
    
    # --- Constant-coefficient polynomial terms ---
    constant_terms = [
        np.ones(n),           # 1 (constant)
        S_t,                  # S
        I_t,                  # I
        S_t ** 2,             # S²
        I_t ** 2,             # I²
        S_t * I_t,            # SI  ← mass-action incidence lives here!
    ]
    constant_labels = ["1", "S", "I", "S²", "I²", "SI"]
    
    if order >= 3:
        constant_terms.extend([
            S_t ** 3,         # S³
            S_t ** 2 * I_t,   # S²I
            S_t * I_t ** 2,   # SI²
            I_t ** 3,         # I³
        ])
        constant_labels.extend(["S³", "S²I", "SI²", "I³"])
    
    # --- Seasonal terms: multiply each polynomial by β(t) ---
    seasonal_terms = [beta_t * term for term in constant_terms]
    seasonal_labels = ["β(t)·" + label for label in constant_labels]
    
    # --- Combine into full library ---
    all_terms = constant_terms + seasonal_terms
    all_labels = constant_labels + seasonal_labels
    
    # Stack columns into matrix Θ
    Theta = np.column_stack(all_terms)
    
    return Theta, all_labels


def get_library_info(order=2):
    """Return the number of terms and their labels for a given polynomial order."""
    if order == 2:
        n_constant = 6
    elif order == 3:
        n_constant = 10
    else:
        raise ValueError(f"Order must be 2 or 3, got {order}")
    
    n_total = n_constant * 2  # constant + seasonal
    return n_total
