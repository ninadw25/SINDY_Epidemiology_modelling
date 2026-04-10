"""
sindy_core.py — The core SINDy algorithm.

This implements sparsifyDynamics() — the heart of the entire paper.

The algorithm:
    1. Start with all candidate terms in the library Θ
    2. Least-squares regression: Ξ = Θ \\ X_next
    3. Compute weighted threshold for each column: λ_w(k) = λ_c / ||Θ_col_k||
    4. Kill any coefficient below its threshold: if |ξ_k| < λ_w(k), set ξ_k = 0
    5. Remove dead columns from Θ, creating smaller Θ'
    6. Repeat 2-5 until convergence (no more eliminations)
    7. Return the sparse coefficient matrix Ξ

The surviving nonzero entries in Ξ ARE the discovered model.
"""

import numpy as np


def sparsify_dynamics(Theta, X_next, lambda_c, max_iter=50):
    """
    The core SINDy sparse regression algorithm.
    
    Equivalent to MATLAB's sparsifyDynamics() function.
    
    This is an iterative thresholding method:
    - Fit all terms via least squares
    - Kill terms with small coefficients (below weighted threshold)
    - Refit surviving terms
    - Repeat until convergence
    
    Parameters
    ----------
    Theta : array of shape (n_timesteps, n_terms)
        The function library matrix. Each column is a candidate term.
    X_next : array of shape (n_timesteps, n_equations)
        The target: X(t+1) for each equation (S and I equations).
    lambda_c : float
        The constant sparsity threshold (the "sparsity knob").
        This gets scaled per-column by the weighted thresholding.
    max_iter : int
        Maximum number of prune-refit iterations.
    
    Returns
    -------
    Xi : array of shape (n_terms, n_equations)
        The sparse coefficient matrix. Nonzero entries = discovered model terms.
    active_mask : array of bool, shape (n_terms,)
        Which library terms survived (True = active, False = eliminated).
    """
    n_terms = Theta.shape[1]
    n_eqs = X_next.shape[1] if X_next.ndim > 1 else 1
    
    if X_next.ndim == 1:
        X_next = X_next.reshape(-1, 1)
    
    # --- Compute weighted thresholds (Equation 21) ---
    # λ_w(k) = λ_c / ||Θ_column_k||
    # This ensures fair treatment of columns with different magnitudes
    column_norms = np.linalg.norm(Theta, axis=0)
    column_norms[column_norms < 1e-10] = 1e-10  # avoid division by zero
    lambda_weighted = lambda_c / column_norms  # shape: (n_terms,)
    
    # --- Initialize: all terms active ---
    active = np.ones(n_terms, dtype=bool)
    Xi = np.zeros((n_terms, n_eqs))
    
    for iteration in range(max_iter):
        # Which columns are still alive?
        active_idx = np.where(active)[0]
        
        if len(active_idx) == 0:
            break  # everything was eliminated — pathological case
        
        # --- Step 2: Least-squares regression on active terms ---
        # Ξ_active = Θ_active \ X_next
        Theta_active = Theta[:, active_idx]
        Xi_active, _, _, _ = np.linalg.lstsq(Theta_active, X_next, rcond=None)
        
        # Store back into full Xi matrix
        Xi[:] = 0
        Xi[active_idx, :] = Xi_active
        
        # --- Step 3-4: Prune terms below weighted threshold ---
        # For each active term, check if |coefficient| < λ_w(k)
        # If so, kill it (set to zero and mark as inactive)
        changed = False
        for k in active_idx:
            # Check across all equations: kill if small in ALL equations
            max_coef = np.max(np.abs(Xi[k, :]))
            if max_coef < lambda_weighted[k]:
                Xi[k, :] = 0.0
                active[k] = False
                changed = True
        
        # --- Step 5: Check convergence ---
        if not changed:
            break  # No terms eliminated this round — we've converged
    
    return Xi, active


def compute_sparsity_index(Xi, n_total_terms):
    """
    Compute the sparsity index r (from the paper).
    
    r = 1 − ||Ξ||₀ / ||Θ||₀
    
    r = 0: every term survived (dense, likely overfitting)
    r = 1: everything was eliminated (too sparse)
    Good models typically have r = 0.25 to 0.7
    
    Parameters
    ----------
    Xi : array — the coefficient matrix from sparsify_dynamics
    n_total_terms : int — total number of terms in the original library
    
    Returns
    -------
    r : float — sparsity index
    """
    n_nonzero = np.sum(np.any(np.abs(Xi) > 1e-10, axis=1))
    r = 1.0 - n_nonzero / n_total_terms
    return r


def print_discovered_model(Xi, labels, equation_names=("S equation", "I equation")):
    """
    Pretty-print the discovered model coefficients.
    
    Parameters
    ----------
    Xi : array of shape (n_terms, n_equations)
    labels : list of str — term names from function_library
    equation_names : tuple of str — names for each equation
    """
    print("\n" + "=" * 70)
    print("DISCOVERED MODEL COEFFICIENTS")
    print("=" * 70)
    
    n_eqs = Xi.shape[1]
    
    # Header
    header = f"{'Term':<15}"
    for name in equation_names[:n_eqs]:
        header += f"  {name:>15}"
    print(header)
    print("-" * 70)
    
    # Each term
    for i, label in enumerate(labels):
        row = f"{label:<15}"
        is_zero = True
        for j in range(n_eqs):
            val = Xi[i, j]
            if abs(val) < 1e-10:
                row += f"  {'0':>15}"
            else:
                row += f"  {val:>15.4f}"
                is_zero = False
        
        # Highlight important terms
        marker = ""
        if not is_zero and "SI" in label and "I²" not in label and "S²" not in label:
            marker = "  ← MASS ACTION"
        elif not is_zero and "β(t)·SI" in label:
            marker = "  ← SEASONAL FORCING"
        
        if not is_zero:
            print(f"\033[1m{row}{marker}\033[0m")  # bold
        else:
            print(f"\033[90m{row}\033[0m")  # gray
    
    # Summary
    n_survived = np.sum(np.any(np.abs(Xi) > 1e-10, axis=1))
    n_total = len(labels)
    r = compute_sparsity_index(Xi, n_total)
    print("-" * 70)
    print(f"Terms survived: {n_survived}/{n_total} | Sparsity index r = {r:.2f}")
    print("=" * 70)
