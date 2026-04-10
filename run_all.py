"""
run_all.py — Entry point: runs the complete SINDy pipeline.

Usage:
    python run_all.py              # Run all three diseases
    python run_all.py measles      # Run measles only
    python run_all.py --fast       # Quick run (reduced grid)

Pipeline:
    1. Generate synthetic disease data (or load from CSV)
    2. Preprocess: smooth → incidence→prevalence → susceptible reconstruction
    3. Build function library Θ(X)
    4. Grid search over (S₀, λ, φ) → run SINDy at each point
    5. Select best model via AIC
    6. Simulate discovered model forward
    7. Generate plots and print coefficient tables
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DISEASES, SINDY, OUTPUT_DIR
from src.sindy_core import print_discovered_model
from src.model_selection import grid_search
from src.psd_analysis import compute_psd, compute_aic_psd
from src.visualization import (
    plot_time_series_comparison,
    plot_coefficients,
    plot_grid_search,
    plot_psd_comparison,
)


def load_or_generate_data(disease_name):
    """Load data from CSV, or generate if not found."""
    filepath = os.path.join("data", f"{disease_name}.csv")
    
    if not os.path.exists(filepath):
        print(f"  Data file not found. Generating synthetic data...")
        from data.generate_data import generate_all_data
        generate_all_data()
    
    df = pd.read_csv(filepath)
    return df


def run_disease(disease_name, fast=False):
    """
    Run the complete SINDy pipeline for one disease.
    
    Parameters
    ----------
    disease_name : str — "measles", "chickenpox", or "rubella"
    fast : bool — use reduced grid for quick testing
    """
    print(f"\n{'='*70}")
    print(f"  RUNNING SINDy FOR: {disease_name.upper()}")
    print(f"{'='*70}")
    
    disease_cfg = DISEASES[disease_name]
    
    # ===== STAGE 1: Load data =====
    print("\n[Stage 1] Loading data...")
    df = load_or_generate_data(disease_name)
    cases = df["cases_incidence"].values.astype(float)
    births = df["births"].values.astype(float)
    population = int(df["population"].iloc[0])
    print(f"  Loaded {len(cases)} weeks of data, population = {population:,}")
    
    # ===== STAGE 2-5: Grid search (preprocessing + library + SINDy inside) =====
    print("\n[Stage 2-5] Preprocessing + Grid search + SINDy...")
    
    if fast:
        S0_range = np.linspace(0.06, 0.12, 5)
        lambda_range = np.logspace(-3.5, -1.5, 5)
        phi_range = np.arange(0, 52, 8.0)
    else:
        S0_range = SINDY["S0_range"]
        lambda_range = SINDY["lambda_range"]
        phi_range = SINDY["phi_range"]
    
    best, all_results = grid_search(
        cases=cases,
        births=births,
        population=population,
        D_i=disease_cfg["D_i"],
        p=disease_cfg["p"],
        L=disease_cfg["L"],
        S0_range=S0_range,
        lambda_range=lambda_range,
        phi_range=phi_range,
        library_order=SINDY["library_order"],
        verbose=True,
    )
    
    if best is None:
        print(f"\n  ✗ No valid model found for {disease_name}. Try adjusting grid ranges.")
        return None
    
    # ===== STAGE 6: Print discovered model =====
    print(f"\n[Stage 6] Discovered model for {disease_name}:")
    print_discovered_model(best["Xi"], best["labels"])
    
    # ===== STAGE 7: Generate plots =====
    print(f"\n[Stage 7] Generating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Time series comparison
    plot_time_series_comparison(
        best["S_t"], best["I_t"],
        best["S_sim"], best["I_sim"],
        disease_name, result_info=best
    )
    
    # Coefficient bar chart
    plot_coefficients(best["Xi"], best["labels"], disease_name)
    
    # Grid search heatmap
    plot_grid_search(all_results, disease_name)
    
    # PSD comparison
    try:
        n_compare = min(len(best["I_t"]), len(best["I_sim"]))
        freq_d, psd_d = compute_psd(best["I_t"][:n_compare])
        freq_m, psd_m = compute_psd(best["I_sim"][:n_compare])
        plot_psd_comparison(freq_d, psd_d, freq_m, psd_m, disease_name)
    except Exception as e:
        print(f"  PSD plot skipped: {e}")
    
    print(f"\n  ✓ {disease_name.title()} complete. Results in '{OUTPUT_DIR}/' folder.")
    return best


def main():
    """Main entry point."""
    args = sys.argv[1:]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]
    
    if fast:
        print("⚡ FAST MODE: Using reduced grid for quick testing")
    
    diseases_to_run = args if args else ["measles", "chickenpox", "rubella"]
    
    results = {}
    for disease in diseases_to_run:
        if disease not in DISEASES:
            print(f"Unknown disease: {disease}. Choose from: {list(DISEASES.keys())}")
            continue
        results[disease] = run_disease(disease, fast=fast)
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, res in results.items():
        if res is None:
            print(f"  {name:>12}: FAILED")
        else:
            print(f"  {name:>12}: AIC={res['aic']:.1f}, "
                  f"sparsity r={res['sparsity']:.2f}, "
                  f"params={res['n_params']}, "
                  f"S₀={res['S0']:.4f}, λ={res['lambda_c']:.5f}")
    
    print(f"\n  All plots saved to '{OUTPUT_DIR}/' folder.")
    print(f"  Done!")


if __name__ == "__main__":
    main()
