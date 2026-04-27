"""
Diagnostic: why don't we recover paper-magnitude coefficients?

Hypothesis chain to test:
1. Plain OLS (no thresholding) -- do we even get paper-like coefficients
   without sparsification? If YES, then sparsifyDynamics is broken.
   If NO, then preprocessing or library is different.
2. Sparsify with paper's exact (S0=0.11286, lambda=0.00517, phi from
   the best phi we find for that point) -- does that reproduce paper?
3. Fit X(t+1) - X(t) (the increment) instead of X(t+1) directly --
   does that recover the paper's negative I-coefficient?
4. Build library with reduced-amplitude beta(t) = 1 + 0.25*cos(...)
   instead of 1 + cos(...) -- does that recover paper-magnitude
   beta(t)*SI?

Also writes a JSON summary of all four fits so update_report.py can
embed them in the docx without rerunning this script.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np

from src.config import DISEASES
from src.preprocessing import preprocess_disease
from src.function_library import build_library
from src.sindy_core import sparsify_dynamics
from run_original_data import load_measles_data


def main():
    cfg = DISEASES["measles"]
    cases, births, population, _ = load_measles_data()

    # Paper says best is S0=0.11286, lambda=0.00517 (Fig. 3 caption).
    S0 = 0.11286
    LAM = 0.00517

    S_t, I_t, _, alpha = preprocess_disease(
        cases, births, population, cfg["D_i"], cfg["p"], cfg["L"],
        S0_fraction=S0,
    )
    print(f"\nPreprocessing summary:")
    print(f"  Mean S_t = {S_t.mean():.4f}, range [{S_t.min():.4f}, {S_t.max():.4f}]")
    print(f"  Mean I_t = {I_t.mean():.6f}, range [{I_t.min():.6f}, {I_t.max():.6f}]")
    print(f"  Reporting rate alpha = {alpha:.4f}")

    t_weeks = np.arange(len(S_t), dtype=float)
    X_next = np.column_stack([S_t[1:], I_t[1:]])

    # The paper does not state phi, so try a few values and pick the one
    # whose SINDy fit assigns the largest coefficient to beta(t)*SI.
    print("\nSweep over phi (with paper's S0, lambda):")
    best_phi, best_betaSI = None, -np.inf
    for phi in np.arange(0, 52, 1.0):
        Theta, labels = build_library(S_t, I_t, t_weeks, phi, order=2)
        Theta_t = Theta[:-1, :]
        Xi, _ = sparsify_dynamics(Theta_t, X_next, LAM)
        k_betaSI = labels.index("β(t)·SI")
        v = float(Xi[k_betaSI, 1])
        if v > best_betaSI:
            best_betaSI = v
            best_phi = phi
        if phi % 4 == 0:
            print(f"  phi={phi:>4.1f}: beta(t)*SI = {v:+.3f}")

    print(f"\nBest phi (max beta*SI): {best_phi}, beta*SI = {best_betaSI:.3f}")

    # Now inspect that fit in detail
    Theta, labels = build_library(S_t, I_t, t_weeks, best_phi, order=2)
    Theta_t = Theta[:-1, :]

    print(f"\nColumn norms of Theta (first row = label, second = ||col||_2):")
    norms = np.linalg.norm(Theta_t, axis=0)
    for lab, n in zip(labels, norms):
        print(f"  {lab:<12} ||col||={n:.5e}")
    cond = np.linalg.cond(Theta_t)
    print(f"\nCondition number of Theta: {cond:.3e}")

    # Plain OLS (no thresholding) -- does THIS give paper magnitudes?
    print("\n=== Plain OLS (no sparsification) ===")
    Xi_ols, *_ = np.linalg.lstsq(Theta_t, X_next, rcond=None)
    print(f"\n{'Term':<14} {'I-eq coef (OLS)':>18}  {'Paper':>10}")
    paper_I = {
        "1": 0.002, "S": -0.037, "I": -1.554, "S²": 0.139, "I²": 0.0,
        "SI": 20.618, "β(t)": -0.013, "β(t)·S": 0.2, "β(t)·I": 0.0,
        "β(t)·S²": -0.0779, "β(t)·SI": 26.409, "β(t)·I²": 0.0,
    }
    for lab, c in zip(labels, Xi_ols[:, 1]):
        ref = paper_I.get(lab, "")
        ref_s = f"{ref:+.3f}" if isinstance(ref, float) else ""
        print(f"  {lab:<12} {c:>+18.4f}  {ref_s:>10}")

    # Run sparsify with paper's exact lambda
    print(f"\n=== sparsify_dynamics with lambda={LAM} ===")
    Xi_sp, _ = sparsify_dynamics(Theta_t, X_next, LAM)
    n_active = int(np.sum(np.any(np.abs(Xi_sp) > 1e-10, axis=1)))
    print(f"Active terms: {n_active}/12")
    for lab, c in zip(labels, Xi_sp[:, 1]):
        ref = paper_I.get(lab, "")
        ref_s = f"{ref:+.3f}" if isinstance(ref, float) else ""
        print(f"  {lab:<12} {c:>+18.4f}  {ref_s:>10}")

    # ============================================================
    # CRITICAL TEST: fit the INCREMENT X(t+1) - X(t) instead of X(t+1).
    # The paper's coefficient I = -1.554 in the I-eq is mathematically
    # impossible for a one-step-ahead direct fit (it would make I go
    # negative each step). It IS plausible if the regression target is
    # the increment delta_X = X(t+1) - X(t), where the I-coef encodes
    # (beta*S - gamma - mu).
    # ============================================================
    print("\n" + "=" * 70)
    print("=== HYPOTHESIS: paper fits delta_X = X(t+1) - X(t) ===")
    print("=" * 70)
    X_curr = np.column_stack([S_t[:-1], I_t[:-1]])
    delta_X = X_next - X_curr

    print(f"\n  delta_I scale: mean={delta_X[:, 1].mean():.6e}, "
          f"std={delta_X[:, 1].std():.6e}")
    print(f"\n=== Plain OLS on delta_X ===")
    Xi_delta, *_ = np.linalg.lstsq(Theta_t, delta_X, rcond=None)
    print(f"\n{'Term':<14} {'I-eq coef (delta-OLS)':>22}  {'Paper':>10}")
    for lab, c in zip(labels, Xi_delta[:, 1]):
        ref = paper_I.get(lab, "")
        ref_s = f"{ref:+.3f}" if isinstance(ref, float) else ""
        marker = "  <-- MATCH" if (
            isinstance(ref, float)
            and abs(ref) > 1e-3
            and abs(c) > 0.3 * abs(ref)
            and (ref > 0) == (c > 0)
        ) else ""
        print(f"  {lab:<12} {c:>+22.4f}  {ref_s:>10}{marker}")

    print(f"\n=== sparsify_dynamics on delta_X with lambda={LAM} ===")
    Xi_delta_sp, _ = sparsify_dynamics(Theta_t, delta_X, LAM)
    n_active = int(np.sum(np.any(np.abs(Xi_delta_sp) > 1e-10, axis=1)))
    print(f"Active terms: {n_active}/12")
    for lab, c in zip(labels, Xi_delta_sp[:, 1]):
        ref = paper_I.get(lab, "")
        ref_s = f"{ref:+.3f}" if isinstance(ref, float) else ""
        print(f"  {lab:<12} {c:>+22.4f}  {ref_s:>10}")

    # ============================================================
    # HYPOTHESIS 3: paper's seasonal forcing has small amplitude
    # beta(t) = 1 + 0.25*cos(...) instead of our 1 + cos(...).
    # Build a library with reduced-amplitude beta and refit.
    # ============================================================
    print("\n" + "=" * 70)
    print("=== HYPOTHESIS: paper's beta(t) = 1 + 0.25*cos(...) ===")
    print("=" * 70)
    beta_low = 1.0 + 0.25 * np.cos(2.0 * np.pi * t_weeks / 52.0 - best_phi)
    # Manually rebuild the library with reduced-amplitude beta
    n = len(S_t)
    base_terms = [np.ones(n), S_t, I_t, S_t**2, I_t**2, S_t * I_t]
    base_lbl = ["1", "S", "I", "S²", "I²", "SI"]
    seas_terms = [beta_low * b for b in base_terms]
    seas_lbl = ["β(t)·" + b for b in base_lbl]
    Theta_low = np.column_stack(base_terms + seas_terms)[:-1]
    labels_low = base_lbl + seas_lbl

    print(f"\n  ||beta(t)*SI|| = {np.linalg.norm(Theta_low[:, labels_low.index('β(t)·SI')]):.5e} "
          f"(was {np.linalg.norm(Theta_t[:, labels.index('β(t)·SI')]):.5e})")

    Xi_low, *_ = np.linalg.lstsq(Theta_low, X_next, rcond=None)
    print(f"\n=== Plain OLS with low-amplitude beta ===")
    print(f"{'Term':<14} {'I-eq coef':>14}  {'Paper':>10}")
    for lab, c in zip(labels_low, Xi_low[:, 1]):
        ref = paper_I.get(lab, "")
        ref_s = f"{ref:+.3f}" if isinstance(ref, float) else ""
        marker = "  <-- MATCH" if (
            isinstance(ref, float)
            and abs(ref) > 1e-3
            and abs(c) > 0.3 * abs(ref)
            and (ref > 0) == (c > 0)
        ) else ""
        print(f"  {lab:<12} {c:>+14.4f}  {ref_s:>10}{marker}")

    # ============================================================
    # Persist a JSON summary for update_report.py to embed.
    # ============================================================
    out_path = os.path.join("outputs_real_data", "magnitude_diagnostic.json")
    payload = {
        "disease": "measles",
        "S0_paper": S0,
        "lambda_paper": LAM,
        "phi_chosen": float(best_phi),
        "preprocessing": {
            "S_t_mean": float(S_t.mean()),
            "S_t_min": float(S_t.min()),
            "S_t_max": float(S_t.max()),
            "I_t_mean": float(I_t.mean()),
            "I_t_min": float(I_t.min()),
            "I_t_max": float(I_t.max()),
            "alpha_reporting_rate": float(alpha),
        },
        "theta_diagnostic": {
            "condition_number": float(cond),
            "column_norms": {lab: float(n) for lab, n in zip(labels, norms)},
        },
        "paper_reference_I_eq": paper_I,
        "fits": {
            "ols_X_next": {lab: float(c) for lab, c in zip(labels, Xi_ols[:, 1])},
            "sparsify_X_next": {lab: float(c) for lab, c in zip(labels, Xi_sp[:, 1])},
            "ols_delta_X": {lab: float(c) for lab, c in zip(labels, Xi_delta[:, 1])},
            "sparsify_delta_X": {lab: float(c)
                                  for lab, c in zip(labels, Xi_delta_sp[:, 1])},
            "ols_low_beta": {lab: float(c) for lab, c in zip(labels_low, Xi_low[:, 1])},
        },
        "headline_term_recovery": _summarise_recovery(
            paper_I, Xi_ols, Xi_sp, Xi_delta, Xi_delta_sp, Xi_low,
            labels, labels_low),
    }
    os.makedirs("outputs_real_data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


def _summarise_recovery(paper_I, Xi_ols, Xi_sp, Xi_delta, Xi_delta_sp, Xi_low,
                        labels, labels_low):
    """For the 5 headline terms, report recovery fraction (ours / paper) per fit."""
    summary = {}
    headline = ("SI", "β(t)·SI", "S²", "I", "β(t)·I²")
    for term in headline:
        ref = paper_I.get(term, 0.0)
        if abs(ref) < 1e-3:
            continue
        row = {"paper": ref}
        for name, Xi, lab_list in [
            ("ols_X_next", Xi_ols, labels),
            ("sparsify_X_next", Xi_sp, labels),
            ("ols_delta_X", Xi_delta, labels),
            ("sparsify_delta_X", Xi_delta_sp, labels),
            ("ols_low_beta", Xi_low, labels_low),
        ]:
            if term in lab_list:
                k = lab_list.index(term)
                v = float(Xi[k, 1])
                row[name] = v
                row[f"{name}_ratio"] = v / ref if ref != 0 else float("nan")
        summary[term] = row
    return summary


if __name__ == "__main__":
    main()
