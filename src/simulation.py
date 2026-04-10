"""
simulation.py — Simulate the discovered model forward in time.

Once SINDy discovers a coefficient matrix Ξ, we can simulate the model:
    X(t+1) = Θ(X(t)) × Ξ

This is equivalent to MATLAB's SparseGalerkin() function.
"""

import numpy as np
from src.function_library import build_library


def simulate_discovered_model(Xi, S0, I0, t_weeks, phi, order=2):
    """
    Simulate the discovered discrete-time model forward.
    
    At each timestep:
        1. Compute the library Θ(X_t) at the current state
        2. Multiply: X(t+1) = Θ(X_t) × Ξ
        3. Step forward
    
    This is equivalent to:
        S(t+1) = sum of (coefficient_k × term_k evaluated at S_t, I_t)
        I(t+1) = sum of (coefficient_k × term_k evaluated at S_t, I_t)
    
    Parameters
    ----------
    Xi : array of shape (n_terms, 2) — discovered coefficient matrix
    S0 : float — initial susceptible fraction
    I0 : float — initial infectious prevalence
    t_weeks : array — time vector in weeks
    phi : float — seasonal phase shift
    order : int — polynomial order used in library
    
    Returns
    -------
    S_sim : array — simulated susceptible time series
    I_sim : array — simulated infectious time series
    """
    n = len(t_weeks)
    S_sim = np.zeros(n)
    I_sim = np.zeros(n)
    
    S_sim[0] = S0
    I_sim[0] = I0
    
    for t in range(n - 1):
        # Build library at the CURRENT state (single timestep)
        S_curr = np.array([S_sim[t]])
        I_curr = np.array([I_sim[t]])
        t_curr = np.array([float(t_weeks[t])])
        
        Theta_t, _ = build_library(S_curr, I_curr, t_curr, phi, order)
        # Theta_t shape: (1, n_terms)
        
        # X(t+1) = Θ(X_t) × Ξ
        next_state = Theta_t @ Xi  # shape: (1, 2)
        
        S_sim[t + 1] = next_state[0, 0]
        I_sim[t + 1] = next_state[0, 1]
        
        # Clamp to prevent runaway dynamics
        S_sim[t + 1] = np.clip(S_sim[t + 1], 0, 1)
        I_sim[t + 1] = np.clip(I_sim[t + 1], 0, 0.1)  # prevalence can't exceed 10%
    
    return S_sim, I_sim
