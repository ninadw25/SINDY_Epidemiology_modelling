"""
regime_shift.py - Out-of-sample regime-shift test (paper Figs. 8-9).

The paper's strongest claim is that a SINDy model discovered on
pre-vaccine measles data (1948-1967) can predict the empirically
observed regime shift to a noisy annual cycle in the vaccine era
(1968-1988) just by reducing the susceptible-recharge rate.

Concretely: in Fig. 9 the S coefficient of the S equation drops from
0.606 to 0.317 (a multiplicative factor of ~0.523) representing
vaccination + falling birth rate. Under that perturbation the model
spontaneously shifts from biennial to noisy annual.

This module implements that test in our pipeline. It:

    1. Takes any discovered Xi + labels (baseline or ensemble).
    2. Multiplies the S-equation entries for the "1" and "S" columns
       by a perturbation factor.
    3. Forward-simulates the perturbed model.
    4. Computes the power spectrum and reports the dominant period.

A small assertion helper checks that:

    * The unperturbed model recovers the disease's expected period
      (within the configured tolerance).
    * Some perturbation factor < 1 produces a measurable period shift
      OR amplitude collapse (regime shift detected).
"""

from typing import Optional

import numpy as np

from src.config import REGIME_SHIFT
from src.psd_analysis import compute_psd
from src.simulation import simulate_discovered_model


_EPS = 1e-10


def perturb_susceptible_dynamics(Xi, labels, factor):
    """
    Reduce the S-equation "1" and "S" columns by `factor`.

    Why these two columns?
    ----------------------
    In the paper the "S" coefficient of the S equation captures the
    effective per-capita recharge rate of the susceptible pool. Cutting
    it (and the constant intercept) is the cleanest mathematical proxy
    for vaccination + falling birth rate, since it leaves the
    transmission terms (SI, beta(t)*SI) -- which encode invariant
    contact-process biology -- untouched.

    Parameters
    ----------
    Xi : (p, 2) array -- discovered coefficient matrix.
    labels : list[str] -- term names (column 0 = "1", column 1 = "S").
    factor : float -- multiplier applied to the S-equation entries
        of the "1" and "S" terms. factor=1.0 leaves the model untouched.

    Returns
    -------
    Xi_perturbed : (p, 2) array -- a copy with the two entries scaled.
    """
    Xi_perturbed = np.array(Xi, copy=True)

    for term in ("1", "S"):
        if term in labels:
            k = labels.index(term)
            Xi_perturbed[k, 0] = Xi[k, 0] * factor

    return Xi_perturbed


def _dominant_period_years(time_series, fs=52.0, min_period_weeks=8.0):
    """
    Locate the dominant cycle period (in years) of a time series.

    Uses the same PSD pipeline as psd_analysis.compute_psd. We ignore
    the DC bin and any frequency higher than 1/min_period_weeks (so
    measurement noise at week-scale frequencies cannot win the argmax).

    Returns NaN if the series is degenerate (all zeros / NaN).
    """
    if len(time_series) < 32 or not np.any(np.isfinite(time_series)):
        return float("nan")
    if np.std(time_series) < 1e-12:
        return float("nan")

    freqs, psd = compute_psd(time_series, fs=fs)
    # frequency in cycles/year; a min_period_weeks weeks => max freq
    # = 52 / min_period_weeks cycles/year.
    max_freq = 52.0 / min_period_weeks
    valid = (freqs > 1e-6) & (freqs <= max_freq)
    if not np.any(valid):
        return float("nan")
    f_v = freqs[valid]
    p_v = psd[valid]
    if not np.any(np.isfinite(p_v)) or np.max(p_v) <= 0:
        return float("nan")
    f_peak = f_v[np.argmax(p_v)]
    if f_peak <= 0:
        return float("nan")
    return float(1.0 / f_peak)  # years


def _amplitude_summary(time_series):
    """Return (range, std) of the simulated I trajectory after burn-in."""
    if len(time_series) < 16:
        return (0.0, 0.0)
    burn = max(52, len(time_series) // 5)
    tail = time_series[burn:]
    rng_v = float(np.nanmax(tail) - np.nanmin(tail))
    std_v = float(np.nanstd(tail))
    return (rng_v, std_v)


def run_regime_shift_test(
    best_result,
    perturbation_factors=None,
    forward_steps=None,
    library_order=2,
):
    """
    Forward-simulate the discovered model under a sequence of S-equation
    perturbations and extract the dominant cycle period of each.

    Parameters
    ----------
    best_result : dict -- output of grid_search or grid_search_ensemble.
        Required keys: Xi (or Xi_filtered), labels, S0 (or S_t), phi.
    perturbation_factors : list[float] or None -- defaults to REGIME_SHIFT.
    forward_steps : int or None -- weeks to simulate (default ~10 years).
    library_order : int -- must match what built the library.

    Returns
    -------
    regime_results : dict with keys
        factors, S_sims, I_sims, freqs, psds, dominant_period_years,
        amplitude_range, amplitude_std, baseline_period_years,
        regime_shift_detected
    """
    if perturbation_factors is None:
        perturbation_factors = REGIME_SHIFT["perturbation_factors"]
    if forward_steps is None:
        forward_steps = REGIME_SHIFT["forward_steps"]

    Xi_base = best_result.get("Xi_filtered", best_result["Xi"])
    labels = best_result["labels"]
    phi = float(best_result["phi"])

    # Initial conditions: use the start of the reconstructed series.
    if "S_t" in best_result and "I_t" in best_result:
        S0 = float(best_result["S_t"][0])
        I0 = float(best_result["I_t"][0])
    else:
        S0 = float(best_result.get("S0", 0.1))
        I0 = 1e-3

    t_weeks = np.arange(forward_steps, dtype=float)

    factors = list(perturbation_factors)
    S_sims = []
    I_sims = []
    freqs_list = []
    psds_list = []
    periods = []
    amp_ranges = []
    amp_stds = []

    for f in factors:
        Xi_p = perturb_susceptible_dynamics(Xi_base, labels, f)
        try:
            S_sim, I_sim = simulate_discovered_model(
                Xi_p, S0, I0, t_weeks, phi, library_order
            )
        except Exception:
            S_sim = np.full_like(t_weeks, np.nan)
            I_sim = np.full_like(t_weeks, np.nan)

        # PSD on infectious trajectory
        try:
            freqs, psd = compute_psd(I_sim, fs=52.0)
        except Exception:
            freqs = np.array([])
            psd = np.array([])

        period = _dominant_period_years(I_sim)
        rng_v, std_v = _amplitude_summary(I_sim)

        S_sims.append(S_sim)
        I_sims.append(I_sim)
        freqs_list.append(freqs)
        psds_list.append(psd)
        periods.append(period)
        amp_ranges.append(rng_v)
        amp_stds.append(std_v)

    # ----- Detect regime shift (period changes OR amplitude collapses) ----
    baseline_idx = int(np.argmin(np.abs(np.asarray(factors) - 1.0)))
    baseline_period = periods[baseline_idx]
    baseline_amp = amp_stds[baseline_idx]
    regime_shift_detected = False
    for i, f in enumerate(factors):
        if i == baseline_idx or f >= 1.0:
            continue
        # Either the period drifted by > 0.3 yr from baseline ...
        if (np.isfinite(baseline_period) and np.isfinite(periods[i])
                and abs(periods[i] - baseline_period) > 0.3):
            regime_shift_detected = True
            break
        # ... or the amplitude collapsed by 50%+
        if baseline_amp > 1e-9 and amp_stds[i] < 0.5 * baseline_amp:
            regime_shift_detected = True
            break

    return {
        "factors": factors,
        "S_sims": S_sims,
        "I_sims": I_sims,
        "freqs": freqs_list,
        "psds": psds_list,
        "dominant_period_years": periods,
        "amplitude_range": amp_ranges,
        "amplitude_std": amp_stds,
        "baseline_period_years": baseline_period,
        "regime_shift_detected": regime_shift_detected,
        "phi": phi,
        "S0": S0,
        "I0": I0,
    }


def assert_regime_shift_metrics(regime_results, disease_name):
    """
    Programmatic verification of the regime-shift outcome.

    Returns a dict so the caller can include it in the final report.
    Does NOT raise: each metric is reported pass/fail independently.
    """
    expected = REGIME_SHIFT["expected_periods_years"].get(disease_name)
    tol = REGIME_SHIFT.get("period_tolerance_years", 0.3)

    baseline_period = regime_results.get("baseline_period_years", float("nan"))
    if expected is not None and np.isfinite(baseline_period):
        baseline_period_ok = bool(abs(baseline_period - expected) <= tol)
    else:
        baseline_period_ok = False

    regime_shift_detected = bool(regime_results.get("regime_shift_detected", False))

    details = (f"baseline_period={baseline_period:.2f}yr "
               f"(expected {expected}yr, tol +/-{tol}); "
               f"regime_shift_detected={regime_shift_detected}")

    return {
        "disease": disease_name,
        "baseline_period_years": baseline_period,
        "expected_period_years": expected,
        "baseline_period_ok": baseline_period_ok,
        "regime_shift_detected": regime_shift_detected,
        "details": details,
    }
