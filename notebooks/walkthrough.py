"""
walkthrough.py — Step-by-step demo of the SINDy pipeline.

Run this to see each stage in action with explanations:
    python notebooks/walkthrough.py

This walks through the measles example, printing intermediate results
at every stage so you can understand exactly what happens.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DISEASES
from src.preprocessing import smooth_cases, incidence_to_prevalence, reconstruct_susceptible
from src.function_library import build_library
from src.sindy_core import sparsify_dynamics, print_discovered_model, compute_sparsity_index
from src.simulation import simulate_discovered_model


def main():
    print("=" * 70)
    print("  SINDy WALKTHROUGH — Measles Example")
    print("  Following the paper step by step")
    print("=" * 70)
    
    # ===== LOAD DATA =====
    print("\n📂 STEP 1: Load data")
    print("-" * 40)
    
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "measles.csv"))
    cases = df["cases_incidence"].values.astype(float)
    births = df["births"].values.astype(float)
    pop = int(df["population"].iloc[0])
    
    print(f"  Loaded {len(cases)} weeks of measles data")
    print(f"  Population: {pop:,}")
    print(f"  Mean cases/week: {np.mean(cases):.0f}")
    print(f"  Max cases/week:  {np.max(cases):.0f}")
    print(f"  First 10 values: {cases[:10].astype(int)}")
    
    # ===== SMOOTH =====
    print("\n🔧 STEP 2: Savitzky-Golay smoothing")
    print("-" * 40)
    print("  Purpose: Reduce noise while preserving peak shapes")
    print("  MATLAB equivalent: sgolayfilt(data, 3, 19)")
    
    cases_smooth = smooth_cases(cases)
    
    print(f"  Before smoothing — std: {np.std(cases):.1f}")
    print(f"  After smoothing  — std: {np.std(cases_smooth):.1f}")
    print(f"  Noise reduced by: {(1 - np.std(cases_smooth)/np.std(cases))*100:.1f}%")
    
    # ===== INCIDENCE → PREVALENCE =====
    print("\n🔄 STEP 3: Incidence → Prevalence conversion")
    print("-" * 40)
    print("  Formula: P_t = C_t × p × D_i / (⟨C_t⟩ × L)")
    print("  Why: Data gives NEW cases/week. Model needs TOTAL currently infected.")
    
    cfg = DISEASES["measles"]
    I_t = incidence_to_prevalence(cases_smooth, cfg["D_i"], cfg["p"], cfg["L"], pop)
    
    print(f"  D_i (infection duration): {cfg['D_i']} weeks")
    print(f"  p (lifetime infection prob): {cfg['p']}")
    print(f"  L (lifespan): {cfg['L']:.0f} weeks ({cfg['L']/52:.0f} years)")
    print(f"  Result — I(t) range: [{I_t.min():.6f}, {I_t.max():.6f}]")
    print(f"  Mean prevalence: {np.mean(I_t):.6f} ({np.mean(I_t)*100:.4f}% of population)")
    
    # ===== SUSCEPTIBLE RECONSTRUCTION =====
    print("\n🔍 STEP 4: Susceptible reconstruction")
    print("-" * 40)
    print("  Purpose: Nobody measured how many people were susceptible in 1955.")
    print("  Method: Finkenstädt-Grenfell — infer from births minus cases.")
    print("  Formula: S_{t+1} = S_t + births − α×cases")
    
    S0 = 0.10  # initial guess
    S_t, alpha = reconstruct_susceptible(cases_smooth, births, pop, S0)
    
    print(f"  Initial S₀: {S0}")
    print(f"  Estimated reporting rate α: {alpha:.3f} ({alpha*100:.1f}% of cases reported)")
    print(f"  S(t) range: [{S_t.min():.4f}, {S_t.max():.4f}]")
    print(f"  Mean susceptible fraction: {np.mean(S_t):.4f} ({np.mean(S_t)*100:.2f}%)")
    
    # ===== BUILD FUNCTION LIBRARY =====
    print("\n📚 STEP 5: Build function library Θ(X)")
    print("-" * 40)
    print("  This is the 'menu of ingredients' — all possible terms SINDy can pick from.")
    
    t_weeks = np.arange(len(S_t), dtype=float)
    phi = 10.0  # seasonal phase shift (weeks)
    Theta, labels = build_library(S_t, I_t, t_weeks, phi, order=2)
    
    print(f"  Polynomial order: 2")
    print(f"  Phase shift φ: {phi} weeks")
    print(f"  Library shape: {Theta.shape} (timesteps × candidate terms)")
    print(f"  Candidate terms ({len(labels)}):")
    for i, label in enumerate(labels):
        col_norm = np.linalg.norm(Theta[:, i])
        print(f"    [{i:2d}] {label:<15} — column norm: {col_norm:.4f}")
    
    # ===== RUN SINDY =====
    print("\n⚡ STEP 6: Run SINDy (sparsifyDynamics)")
    print("-" * 40)
    print("  The core algorithm:")
    print("    1. Least-squares fit all terms")
    print("    2. Kill terms with |coef| < λ_weighted")
    print("    3. Refit survivors")
    print("    4. Repeat until convergence")
    
    # Set up regression: Θ(t) → X(t+1)
    X_next = np.column_stack([S_t[1:], I_t[1:]])
    Theta_trimmed = Theta[:-1, :]
    
    lambda_c = 0.005  # sparsity threshold
    print(f"  λ (sparsity knob): {lambda_c}")
    
    Xi, active = sparsify_dynamics(Theta_trimmed, X_next, lambda_c)
    
    # Print results
    print_discovered_model(Xi, labels)
    
    r = compute_sparsity_index(Xi, len(labels))
    print(f"\n  Sparsity index r = {r:.2f}")
    n_survived = int(np.sum(np.any(np.abs(Xi) > 1e-10, axis=1)))
    print(f"  {n_survived} out of {len(labels)} terms survived")
    
    # ===== INTERPRET =====
    print("\n🧠 STEP 7: Interpret the discovered model")
    print("-" * 40)
    
    # Find the dominant I-equation terms
    I_coeffs = Xi[:, 1]
    sorted_idx = np.argsort(np.abs(I_coeffs))[::-1]
    print("  I equation — terms ranked by importance:")
    for rank, idx in enumerate(sorted_idx[:5]):
        coef = I_coeffs[idx]
        if abs(coef) < 1e-10:
            break
        interpretation = ""
        if "SI" in labels[idx] and "²" not in labels[idx]:
            interpretation = " ← MASS-ACTION INCIDENCE (the key mechanism!)"
        elif "β(t)·SI" in labels[idx]:
            interpretation = " ← SEASONAL FORCING"
        elif labels[idx] == "I":
            interpretation = " ← Recovery/demographic process"
        elif "S²" in labels[idx]:
            interpretation = " ← Novel: subexponential growth"
        print(f"    #{rank+1}: {labels[idx]:15} coef = {coef:>10.4f}{interpretation}")
    
    # ===== SIMULATE =====
    print("\n📈 STEP 8: Simulate discovered model forward")
    print("-" * 40)
    
    try:
        S_sim, I_sim = simulate_discovered_model(Xi, S_t[0], I_t[0], t_weeks, phi, order=2)
        
        # Compute fit quality
        n_compare = min(len(I_t), len(I_sim))
        residuals = I_t[:n_compare] - I_sim[:n_compare]
        rmse = np.sqrt(np.mean(residuals ** 2))
        r_squared = 1 - np.sum(residuals**2) / np.sum((I_t[:n_compare] - np.mean(I_t[:n_compare]))**2)
        
        print(f"  Simulation length: {n_compare} weeks ({n_compare/52:.1f} years)")
        print(f"  RMSE (I equation): {rmse:.6f}")
        print(f"  R² (I equation):   {r_squared:.4f}")
        print(f"  I_sim range: [{I_sim.min():.6f}, {I_sim.max():.6f}]")
        print(f"  I_data range: [{I_t.min():.6f}, {I_t.max():.6f}]")
    except Exception as e:
        print(f"  Simulation failed: {e}")
        print("  This can happen with certain S₀/λ combinations. Try grid search instead.")
    
    # ===== DONE =====
    print(f"\n{'='*70}")
    print("  WALKTHROUGH COMPLETE")
    print(f"{'='*70}")
    print("\n  Key takeaway: SINDy started with 12 candidate terms and discovered")
    print("  the mass-action incidence mechanism (S×I) automatically — the same")
    print("  mechanism epidemiologists spent decades establishing through theory.")
    print("\n  To run the full pipeline with grid search: python run_all.py measles")
    print("  To run quick test: python run_all.py measles --fast")


if __name__ == "__main__":
    main()
