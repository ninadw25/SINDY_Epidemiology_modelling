"""
results_io.py - Serialization helpers for paper-vs-reproduction comparison.

The baseline pipeline (run_original_data.py) only emits PNG plots, so the
discovered coefficients live in memory and are lost between runs. This
module turns each grid_search / ensemble_grid_search output into:

    1. A flat CSV row (results_to_record + save_results_csv)
       -> headline coefficients side-by-side with paper values.
    2. A full JSON dump (save_full_xi_json)
       -> every library term, plus inclusion probabilities and IQR for
          the ensemble case.

The paper reference table (load_paper_reference) is hand-transcribed from
Horrocks & Bauch (2020), Figs. 3, 4, 6 -- the I equation values for the
three diseases the paper reports.
"""

import csv
import json
import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Paper ground truth (Horrocks & Bauch, 2020, Sci. Rep. 10:7061)
# ---------------------------------------------------------------------------
# Coefficients are taken from the tables in Figs. 3, 4, 6 of the paper.
# Only the I-equation column is recorded -- this is what the paper itself
# discusses, and S is a reconstructed series so its coefficients are less
# directly meaningful.
#
# Keys match the labels emitted by function_library.build_library so a
# direct lookup is possible.

_PAPER_REFERENCE = {
    "measles": {  # Fig. 3 (baseline, 2nd-order library)
        "S0": 0.11286,
        "lambda_c": 0.00517,
        "sparsity_r": 0.25,
        "method": "baseline_time_series",
        "I_eq": {
            "1": 0.002,
            "S": -0.037,
            "I": -1.554,
            "S²": 0.139,
            "SI": 20.618,
            "I²": 0.0,
            "β(t)": -0.013,
            "β(t)·S": 0.2,
            "β(t)·I": 0.0,
            "β(t)·S²": -0.0779,
            "β(t)·SI": 26.409,
            "β(t)·I²": 0.0,
        },
        "S_eq": {
            "1": 0.025,
            "S": 0.606,
            "I": 1.084,
            "S²": 1.541,
            "SI": -10.295,
            "I²": 0.0,
            "β(t)": -0.009,
            "β(t)·S": 0.146,
            "β(t)·I": -3.122,
            "β(t)·S²": -0.591,
            "β(t)·SI": 26.409,
            "β(t)·I²": 0.0,
        },
        "regime_shift_S_coef_after": 0.317,  # paper Fig. 9 perturbation
    },
    "chickenpox": {  # Fig. 4 (baseline, 2nd-order library) - shows overfit
        "S0": None,  # paper does not state numerically in main text
        "lambda_c": None,
        "sparsity_r": 0.25,
        "method": "baseline_time_series",
        "I_eq": {
            "1": -0.002,
            "S": 0.081,
            "I": 0.953,
            "S²": -0.776,
            "SI": 0.0,
            "I²": 0.0,
            "β(t)": 0.001,
            "β(t)·S": 0.001,
            "β(t)·I": -1.163,
            "β(t)·S²": 0.0,
            "β(t)·SI": 19.24,
            "β(t)·I²": 114.765,  # spurious -- the central failure mode
        },
        "S_eq": {
            "1": 0.0,
            "S": 1.011,
            "I": 1.043,
            "S²": -0.037,
            "SI": -26.601,
            "I²": 0.0,
            "β(t)": 0.0,
            "β(t)·S": -0.024,
            "β(t)·I": 0.215,
            "β(t)·S²": 0.267,
            "β(t)·SI": -3.749,
            "β(t)·I²": 0.0,
        },
    },
    "rubella": {  # Fig. 6 (PSD-fit, 2nd-order library) -- the recovery
        "S0": None,
        "lambda_c": None,
        "sparsity_r": None,
        "method": "psd_fit",
        "I_eq": {
            "1": 0.0033,
            "S": -0.0452,
            "I": -2.2542,
            "S²": 0.1547,
            "SI": 23.7858,
            "I²": 0.0,
            "β(t)": 0.0,
            "β(t)·S": 0.0,
            "β(t)·I": -3.5303,
            "β(t)·S²": 0.0,
            "β(t)·SI": 25.6070,
            "β(t)·I²": 0.0,
        },
        "S_eq": {
            "1": 0.0022,
            "S": 0.9718,
            "I": -0.3461,
            "S²": 0.0968,
            "SI": 0.0,
            "I²": 55.9012,
            "β(t)": -0.0015,
            "β(t)·S": 0.0204,
            "β(t)·I": -0.6015,
            "β(t)·S²": -0.0690,
            "β(t)·SI": 5.5582,
            "β(t)·I²": 0.0,
        },
    },
}


def load_paper_reference():
    """Return the hand-transcribed paper reference table (deep-copy safe dict)."""
    return json.loads(json.dumps(_PAPER_REFERENCE))


# ---------------------------------------------------------------------------
# Headline term names -- these get their own columns in results.csv
# ---------------------------------------------------------------------------
HEADLINE_TERMS = ["SI", "β(t)·SI", "S²", "I²", "β(t)·I²"]


def _safe_lookup(coef_dict, label, default=0.0):
    return coef_dict.get(label, default) if coef_dict is not None else default


def _coef_at_label(Xi, labels, label, eq_idx):
    """Return Xi[k, eq_idx] where labels[k] == label, or 0.0 if absent."""
    if label not in labels:
        return 0.0
    k = labels.index(label)
    return float(Xi[k, eq_idx])


def result_to_record(disease, method, best, paper_reference=None):
    """
    Flatten a `best` result dict (from grid_search or grid_search_ensemble)
    into a single CSV-ready record.

    Parameters
    ----------
    disease : str -- "measles" / "chickenpox" / "rubella"
    method : str  -- "baseline_sindy" / "ensemble_sindy" / "paper"
    best : dict   -- contains Xi, labels, S0, lambda_c, phi, aic, sparsity,
                     n_params, alpha; for ensemble also Xi_median, Xi_q25,
                     Xi_q75, inclusion_prob.
    paper_reference : optional dict -- output of load_paper_reference()
                                       to fill paper-side columns.

    Returns
    -------
    record : dict -- one row, stable column order via OrderedDict semantics.
    """
    record = {
        "disease": disease,
        "method": method,
        "S0": float(best.get("S0", float("nan"))),
        "lambda_c": float(best.get("lambda_c", float("nan"))),
        "phi": float(best.get("phi", float("nan"))),
        "aic": float(best.get("aic", float("nan"))),
        "sparsity_r": float(best.get("sparsity", float("nan"))),
        "n_params": int(best.get("n_params", 0)),
        "alpha_reporting_rate": float(best.get("alpha", float("nan"))),
    }

    # Coefficients in the I equation (the one with observable data)
    Xi = best.get("Xi_median") if "Xi_median" in best else best.get("Xi")
    labels = best["labels"]
    for term in HEADLINE_TERMS:
        record[f"I_eq_{term}"] = _coef_at_label(Xi, labels, term, eq_idx=1)

    # Ensemble-only fields
    if "inclusion_prob" in best:
        for term in HEADLINE_TERMS:
            if term in labels:
                k = labels.index(term)
                record[f"incl_prob_I_eq_{term}"] = float(best["inclusion_prob"][k, 1])
                q25 = best["Xi_q25"][k, 1]
                q75 = best["Xi_q75"][k, 1]
                record[f"iqr_I_eq_{term}"] = float(q75 - q25)
            else:
                record[f"incl_prob_I_eq_{term}"] = 0.0
                record[f"iqr_I_eq_{term}"] = 0.0
    else:
        for term in HEADLINE_TERMS:
            record[f"incl_prob_I_eq_{term}"] = float("nan")
            record[f"iqr_I_eq_{term}"] = float("nan")

    # Paper reference columns -- the comparison story
    if paper_reference is not None and disease in paper_reference:
        ref_I = paper_reference[disease].get("I_eq", {})
        for term in HEADLINE_TERMS:
            record[f"paper_I_eq_{term}"] = float(_safe_lookup(ref_I, term, 0.0))
    else:
        for term in HEADLINE_TERMS:
            record[f"paper_I_eq_{term}"] = float("nan")

    return record


def save_results_csv(records, path):
    """
    Write a list of records (from result_to_record) to CSV. Stable column
    order = union of keys, encountered-first.
    """
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fieldnames = []
    seen = set()
    for r in records:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"  Saved CSV: {path}")


def _to_serializable(obj):
    """Recursively convert numpy types -> Python primitives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def save_full_xi_json(disease, method, best, path):
    """
    Dump the full discovered model -- every library term -- to JSON.

    Includes Xi (or Xi_median for ensemble), labels, hyperparameters,
    and ensemble-specific fields when present.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    payload = {
        "disease": disease,
        "method": method,
        "labels": list(best["labels"]),
        "S0": best.get("S0"),
        "lambda_c": best.get("lambda_c"),
        "phi": best.get("phi"),
        "aic": best.get("aic"),
        "sparsity_r": best.get("sparsity"),
        "n_params": best.get("n_params"),
        "alpha_reporting_rate": best.get("alpha"),
    }

    # Either Xi (baseline) or Xi_median (ensemble) lives at "Xi"
    Xi_to_save = best.get("Xi_median") if "Xi_median" in best else best.get("Xi")
    if Xi_to_save is not None:
        payload["Xi"] = np.asarray(Xi_to_save).tolist()

    # Ensemble-specific
    for k in ("Xi_median", "Xi_q25", "Xi_q75", "inclusion_prob"):
        if k in best and best[k] is not None:
            payload[k] = np.asarray(best[k]).tolist()
    if "Xi_filtered" in best:
        payload["Xi_filtered"] = np.asarray(best["Xi_filtered"]).tolist()
    if "n_bootstrap" in best:
        payload["n_bootstrap"] = int(best["n_bootstrap"])
    if "inclusion_threshold" in best:
        payload["inclusion_threshold"] = float(best["inclusion_threshold"])

    # Sensitivity sweep over inclusion thresholds, if computed
    if "threshold_sensitivity" in best:
        payload["threshold_sensitivity"] = _to_serializable(best["threshold_sensitivity"])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)
    print(f"  Saved JSON: {path}")
