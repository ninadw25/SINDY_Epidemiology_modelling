"""
Quick experiment: does switching to in-sample AIC (paper's original
scoring) recover paper-magnitude coefficients?

Runs ONE disease (measles) on the FULL grid with use_cv_aic=False so
we can compare against the same disease run with use_cv_aic=True.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

# Force UTF-8 stdout for printing the unicode labels.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
from src.config import DISEASES, SINDY
from src.model_selection import grid_search
from src.sindy_core import print_discovered_model
from run_original_data import load_measles_data


def main():
    cfg = DISEASES["measles"]
    cases, births, population, year_fracs = load_measles_data()

    print("\n=== TEST: in-sample AIC (paper's original scoring) ===\n")
    best, _ = grid_search(
        cases=cases, births=births, population=population,
        D_i=cfg["D_i"], p=cfg["p"], L=cfg["L"],
        S0_range=SINDY["S0_range"],
        lambda_range=SINDY["lambda_range"],
        phi_range=SINDY["phi_range"],
        library_order=SINDY["library_order"],
        use_cv_aic=False,    # <-- the key difference
        verbose=True,
    )
    if best is None:
        print("FAILED")
        return

    print_discovered_model(best["Xi"], best["labels"])
    print(f"\n  Best: S0={best['S0']:.5f}, lambda={best['lambda_c']:.5f}, "
          f"phi={best['phi']:.1f}, AIC={best['aic']:.1f}, "
          f"sparsity={best['sparsity']:.2f}")

    print("\n  Paper headline values for measles I-eq (Fig. 3):")
    print("    SI       = +20.618")
    print("    beta*SI  = +26.409")
    print("    S^2      = +0.139")
    print("    I        = -1.554")

    print("\n  Our discovered I-eq coefficients:")
    Xi = best["Xi"]
    labels = best["labels"]
    for term in ("SI", "β(t)·SI", "S²", "I", "I²", "β(t)·I²"):
        if term in labels:
            k = labels.index(term)
            print(f"    {term:<10} = {Xi[k, 1]:+.4f}")


if __name__ == "__main__":
    main()
