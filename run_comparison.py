"""
run_comparison.py - Baseline SINDy vs Ensemble SINDy on real disease data.

This is the endsem deliverable's top-level entry point. For each disease
(measles / chickenpox / rubella) it:

    1. Loads the IIDDA historical data (reuses run_original_data.py).
    2. Runs baseline grid_search (forward-chained CV-AIC).
    3. Runs grid_search_ensemble (E-SINDy, two-pass).
    4. Runs the regime-shift out-of-sample test on the ensemble model.
    5. Saves baseline_*.json + ensemble_*.json + appends rows to
       results.csv with paper-vs-reproduction comparison columns.
    6. Generates the four new plots:
         * <disease>_coefficients_ensemble.png
         * <disease>_inclusion_heatmap.png
         * <disease>_regime_shift.png
         * <disease>_method_comparison.png

CLI
---
    python run_comparison.py                    # all 3 diseases, full grid (~45 min)
    python run_comparison.py --fast             # reduced grid + B=20 (~5 min)
    python run_comparison.py measles            # single disease, full grid
    python run_comparison.py --fast chickenpox  # smoke test for one disease

The baseline pipeline at run_original_data.py is preserved untouched and
remains usable for legacy comparisons.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")

# Console label strings contain unicode (beta, superscripts). cp1252 console
# encoding on Windows raises UnicodeEncodeError; force UTF-8 with replace
# fallback so a print bug never aborts the whole pipeline.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Add project root so 'from src...' works whether run from root or elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DISEASES, ENSEMBLE, SINDY  # noqa: E402
from src.ensemble_sindy import grid_search_ensemble, print_ensemble_model  # noqa: E402
from src.model_selection import grid_search  # noqa: E402
from src.regime_shift import (  # noqa: E402
    assert_regime_shift_metrics,
    run_regime_shift_test,
)
from src.results_io import (  # noqa: E402
    HEADLINE_TERMS,
    load_paper_reference,
    result_to_record,
    save_full_xi_json,
    save_results_csv,
)
from src.sindy_core import print_discovered_model  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_coefficients_with_uncertainty,
    plot_inclusion_heatmap,
    plot_method_comparison,
    plot_regime_shift,
)
from run_original_data import (  # noqa: E402
    load_chickenpox_data,
    load_measles_data,
    load_rubella_data,
)


OUTPUT_DIR = "outputs_real_data"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _grid_for_mode(fast):
    """Return (S0_range, lambda_range, phi_range, n_bootstrap)."""
    if fast:
        S0_range = np.linspace(0.05, 0.13, 6)
        lambda_range = np.logspace(-4, -1, 6)
        phi_range = np.arange(0, 52, 6.0)
        n_bootstrap = ENSEMBLE.get("fast_n_bootstrap", 20)
    else:
        S0_range = SINDY["S0_range"]
        lambda_range = SINDY["lambda_range"]
        phi_range = SINDY["phi_range"]
        n_bootstrap = ENSEMBLE.get("n_bootstrap", 100)
    return S0_range, lambda_range, phi_range, n_bootstrap


def _load_disease(disease_name):
    if disease_name == "measles":
        return load_measles_data()
    if disease_name == "chickenpox":
        return load_chickenpox_data()
    if disease_name == "rubella":
        return load_rubella_data()
    raise ValueError(f"Unknown disease: {disease_name}")


def _run_one_disease(disease_name, fast, paper_reference):
    """
    Run baseline + ensemble + regime-shift for one disease.

    Returns (records, baseline, ensemble, regime, verify) where
    `records` is a 2-element list [baseline_record, ensemble_record].
    """
    print(f"\n{'=' * 72}\n  {disease_name.upper()}\n{'=' * 72}")
    cfg = DISEASES[disease_name]
    cases, births, population, year_fracs = _load_disease(disease_name)

    S0_range, lambda_range, phi_range, n_bootstrap = _grid_for_mode(fast)

    # ----- BASELINE ---------------------------------------------------------
    print(f"\n[{disease_name}] Baseline grid_search (CV-AIC)...")
    baseline, all_results = grid_search(
        cases=cases, births=births, population=population,
        D_i=cfg["D_i"], p=cfg["p"], L=cfg["L"],
        S0_range=S0_range, lambda_range=lambda_range, phi_range=phi_range,
        library_order=SINDY["library_order"],
        verbose=True,
    )
    if baseline is None:
        print(f"[{disease_name}] Baseline failed -- no valid model.")
        return [], None, None, None, None

    print_discovered_model(baseline["Xi"], baseline["labels"])

    # ----- ENSEMBLE ---------------------------------------------------------
    print(f"\n[{disease_name}] Ensemble grid_search_ensemble (B={n_bootstrap})...")
    ensemble, _pass1 = grid_search_ensemble(
        cases=cases, births=births, population=population,
        D_i=cfg["D_i"], p=cfg["p"], L=cfg["L"],
        S0_range=S0_range, lambda_range=lambda_range, phi_range=phi_range,
        library_order=SINDY["library_order"],
        n_bootstrap=n_bootstrap,
        verbose=True,
    )
    if ensemble is None:
        print(f"[{disease_name}] Ensemble failed.")
        return [], baseline, None, None, None

    print_ensemble_model(ensemble)

    # ----- REGIME-SHIFT -----------------------------------------------------
    print(f"\n[{disease_name}] Regime-shift forward simulation...")
    regime = run_regime_shift_test(ensemble, library_order=SINDY["library_order"])
    verify = assert_regime_shift_metrics(regime, disease_name)
    print(f"  Verification: {verify['details']}")

    # ----- EXPORT -----------------------------------------------------------
    base_record = result_to_record(disease_name, "baseline_sindy",
                                   baseline, paper_reference)
    ens_record = result_to_record(disease_name, "ensemble_sindy",
                                  ensemble, paper_reference)

    save_full_xi_json(disease_name, "baseline", baseline,
                      os.path.join(OUTPUT_DIR, f"{disease_name}_baseline.json"))
    save_full_xi_json(disease_name, "ensemble", ensemble,
                      os.path.join(OUTPUT_DIR, f"{disease_name}_ensemble.json"))

    # ----- PLOTS ------------------------------------------------------------
    print(f"\n[{disease_name}] Generating ensemble plots...")
    plot_coefficients_with_uncertainty(
        ensemble["Xi_median"], ensemble["Xi_q25"], ensemble["Xi_q75"],
        ensemble["inclusion_prob"], ensemble["labels"], disease_name,
        inclusion_threshold=ensemble.get("inclusion_threshold", 0.6),
        output_dir=OUTPUT_DIR,
    )
    plot_inclusion_heatmap(
        ensemble["inclusion_prob"], ensemble["labels"], disease_name,
        inclusion_threshold=ensemble.get("inclusion_threshold", 0.6),
        output_dir=OUTPUT_DIR,
    )
    plot_regime_shift(regime, disease_name, output_dir=OUTPUT_DIR)
    plot_method_comparison([base_record, ens_record], disease_name,
                           output_dir=OUTPUT_DIR)

    return [base_record, ens_record], baseline, ensemble, regime, verify


def _print_summary(all_records, all_verifications):
    """Final summary table -- the report's headline output."""
    print(f"\n{'=' * 72}\n  FINAL SUMMARY\n{'=' * 72}")

    print("\n--- Headline I-equation coefficients (paper / baseline / ensemble) ---")
    by_disease = {}
    for r in all_records:
        by_disease.setdefault(r["disease"], {})[r["method"]] = r
    for disease, methods in by_disease.items():
        print(f"\n  {disease}:")
        for term in HEADLINE_TERMS:
            paper_val = (methods.get("baseline_sindy") or
                         methods.get("ensemble_sindy") or {}).get(
                f"paper_I_eq_{term}", float("nan"))
            base_val = methods.get("baseline_sindy", {}).get(
                f"I_eq_{term}", float("nan"))
            ens_val = methods.get("ensemble_sindy", {}).get(
                f"I_eq_{term}", float("nan"))
            ens_p = methods.get("ensemble_sindy", {}).get(
                f"incl_prob_I_eq_{term}", float("nan"))
            ens_iqr = methods.get("ensemble_sindy", {}).get(
                f"iqr_I_eq_{term}", float("nan"))
            print(f"    {term:>10}: paper={paper_val:>+10.4f}  "
                  f"baseline={base_val:>+10.4f}  "
                  f"ensemble={ens_val:>+10.4f}  "
                  f"(IQR={ens_iqr:.3f}, P_incl={ens_p:.2f})")

    print("\n--- Spurious-term suppression (ensemble inclusion < 0.5) ---")
    for r in all_records:
        if r["method"] != "ensemble_sindy":
            continue
        suppressed = []
        for term in HEADLINE_TERMS:
            p = r.get(f"incl_prob_I_eq_{term}", float("nan"))
            if np.isfinite(p) and p < 0.5:
                suppressed.append(f"{term}(P={p:.2f})")
        msg = ", ".join(suppressed) if suppressed else "(none)"
        print(f"    {r['disease']:>12}: {msg}")

    print("\n--- Regime-shift verification ---")
    for v in all_verifications:
        if v is None:
            continue
        marker = "OK" if v["baseline_period_ok"] else "MISS"
        print(f"    {v['disease']:>12}: baseline period {marker}, "
              f"shift detected: {v['regime_shift_detected']}")
        print(f"                  {v['details']}")


def main():
    args = sys.argv[1:]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]

    if fast:
        print("FAST MODE: reduced grid + B=20 bootstraps for quick smoke test.")

    diseases_to_run = args if args else ["measles", "chickenpox", "rubella"]

    # Validate
    for d in diseases_to_run:
        if d not in DISEASES:
            print(f"Unknown disease: {d}. "
                  f"Choose from: {list(DISEASES.keys())}")
            sys.exit(1)

    paper_reference = load_paper_reference()

    print(f"\n{'#' * 72}")
    print(f"#  Baseline SINDy vs Ensemble SINDy")
    print(f"#  Horrocks & Bauch (2020) reproduction with E-SINDy upgrade")
    print(f"#  Fasel et al. (2022), Proc. Roy. Soc. A 478:20210904")
    print(f"{'#' * 72}")

    all_records = []
    all_verifications = []
    for disease in diseases_to_run:
        try:
            records, _baseline, _ensemble, _regime, verify = \
                _run_one_disease(disease, fast, paper_reference)
        except Exception as exc:
            print(f"[{disease}] FAILED with {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            continue
        all_records.extend(records)
        if verify is not None:
            all_verifications.append(verify)

    if all_records:
        save_results_csv(all_records, os.path.join(OUTPUT_DIR, "results.csv"))
    _print_summary(all_records, all_verifications)
    print(f"\n  Outputs written to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
