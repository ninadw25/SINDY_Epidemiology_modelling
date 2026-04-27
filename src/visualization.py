"""
visualization.py — All plotting functions for SINDy results.

Generates:
    1. Time series comparison (SINDy model vs data)
    2. Coefficient bar charts
    3. Parameter grid heatmaps (S₀ vs λ)
    4. Power spectral density comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import OUTPUT_DIR, FIGURE_DPI


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_time_series_comparison(S_data, I_data, S_sim, I_sim,
                                 disease_name, result_info=None, save=True):
    """
    Plot SINDy model predictions vs actual data.
    
    Top panel: Susceptible time series (S)
    Bottom panel: Infectious time series (I)
    
    Parameters
    ----------
    S_data, I_data : arrays — preprocessed data
    S_sim, I_sim : arrays — SINDy model simulation
    disease_name : str — for the title
    result_info : dict — optional info (S0, lambda, phi, etc.)
    save : bool — save to file
    """
    ensure_output_dir()
    
    n = min(len(S_data), len(S_sim))
    weeks = np.arange(n)
    years = weeks / 52.0 + 1948  # approximate year axis
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Susceptible
    ax1.plot(years[:n], S_data[:n], "b-", alpha=0.7, linewidth=1, label="Data")
    ax1.plot(years[:n], S_sim[:n], "r--", alpha=0.8, linewidth=1, label="SINDy")
    ax1.set_ylabel("Susceptible (proportion)")
    ax1.set_title(f"{disease_name.title()} — Susceptible Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Infectious
    ax2.plot(years[:n], I_data[:n], "b-", alpha=0.7, linewidth=1, label="Data")
    ax2.plot(years[:n], I_sim[:n], "r--", alpha=0.8, linewidth=1, label="SINDy")
    ax2.set_ylabel("Infectious (proportion)")
    ax2.set_xlabel("Time (years)")
    ax2.set_title(f"{disease_name.title()} — Infectious Time Series")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add info text
    if result_info:
        info_text = (f"S₀={result_info.get('S0', '?'):.4f}, "
                     f"λ={result_info.get('lambda_c', '?'):.5f}, "
                     f"φ={result_info.get('phi', '?'):.1f}wk, "
                     f"r={result_info.get('sparsity', '?'):.2f}")
        fig.suptitle(info_text, fontsize=10, y=0.02, color="gray")
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_time_series.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    
    plt.close()


def plot_coefficients(Xi, labels, disease_name, save=True):
    """
    Bar chart of discovered model coefficients.
    
    Equivalent to the coefficient tables in Figures 3-6 of the paper,
    visualized as bar charts like Figure 7.
    
    Parameters
    ----------
    Xi : array of shape (n_terms, 2) — coefficient matrix
    labels : list of str — term names
    disease_name : str
    save : bool
    """
    ensure_output_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    eq_names = ["S equation", "I equation"]
    axes = [ax1, ax2]
    colors = ["steelblue", "coral"]
    
    for eq_idx, (ax, name, color) in enumerate(zip(axes, eq_names, colors)):
        coeffs = Xi[:, eq_idx]
        
        # Only show nonzero terms
        nonzero = np.abs(coeffs) > 1e-10
        if not np.any(nonzero):
            ax.text(0.5, 0.5, "All terms eliminated", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(name)
            continue
        
        idx = np.where(nonzero)[0]
        vals = coeffs[idx]
        labs = [labels[i] for i in idx]
        
        bar_colors = [color if v >= 0 else "gray" for v in vals]
        ax.barh(range(len(vals)), vals, color=bar_colors, alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labs, fontsize=9)
        ax.set_xlabel("Coefficient value")
        ax.set_title(name)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="x")
    
    plt.suptitle(f"{disease_name.title()} — Discovered Model Coefficients", fontsize=14)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_coefficients.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    
    plt.close()


def plot_grid_search(all_results, disease_name, save=True):
    """
    Plot the S₀-λ parameter grid showing AIC scores.
    
    Equivalent to the parameter planes in Supplementary Figures 5-15.
    
    Parameters
    ----------
    all_results : list of dicts with S0, lambda_c, aic, sparsity
    disease_name : str
    save : bool
    """
    ensure_output_dir()
    
    if len(all_results) < 4:
        return
    
    S0_vals = np.array([r["S0"] for r in all_results])
    lam_vals = np.array([r["lambda_c"] for r in all_results])
    aic_vals = np.array([r["aic"] for r in all_results])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    scatter = ax.scatter(S0_vals, lam_vals, c=aic_vals, cmap="RdYlGn_r",
                         s=40, alpha=0.8, edgecolors="gray", linewidth=0.3)
    
    ax.set_xlabel("S₀ (initial susceptible fraction)")
    ax.set_ylabel("λ (sparsity threshold)")
    ax.set_yscale("log")
    ax.set_title(f"{disease_name.title()} — AIC across S₀-λ Grid")
    plt.colorbar(scatter, ax=ax, label="AIC (lower = better)")
    
    # Mark the best point
    best_idx = np.argmin(aic_vals)
    ax.scatter([S0_vals[best_idx]], [lam_vals[best_idx]],
               marker="*", s=200, c="black", zorder=5, label="Best")
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_grid_search.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    
    plt.close()


def plot_coefficients_with_uncertainty(
    Xi_median, Xi_q25, Xi_q75, inclusion_prob,
    labels, disease_name, inclusion_threshold=0.6, save=True,
    output_dir=None,
):
    """
    Bar chart of ensemble coefficient medians with IQR error bars.

    Two panels (S equation, I equation). Bar height = median across
    bootstraps; error bars = q25..q75; bar opacity proportional to
    inclusion probability. Terms below `inclusion_threshold` are drawn
    in light gray with hatching to flag them as filtered out.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    eq_names = ["S equation", "I equation"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for eq_idx, (ax, name) in enumerate(zip(axes, eq_names)):
        med = Xi_median[:, eq_idx]
        q25 = Xi_q25[:, eq_idx]
        q75 = Xi_q75[:, eq_idx]
        p_incl = inclusion_prob[:, eq_idx]

        # Show ANY term that has nonzero median or non-trivial inclusion
        show = (np.abs(med) > 1e-10) | (p_incl > 0.05)
        if not np.any(show):
            ax.text(0.5, 0.5, "No active terms", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(name)
            continue

        idx = np.where(show)[0]
        order = idx[np.argsort(med[idx])]
        med_o = med[order]
        q25_o = q25[order]
        q75_o = q75[order]
        p_o = p_incl[order]
        labs = [labels[i] for i in order]

        ypos = np.arange(len(order))
        # Asymmetric error bars: distance from median to each quantile
        err_low = np.maximum(med_o - q25_o, 0.0)
        err_high = np.maximum(q75_o - med_o, 0.0)

        for i, (m, lo, hi, p) in enumerate(zip(med_o, err_low, err_high, p_o)):
            survives = p >= inclusion_threshold
            if survives:
                color = "steelblue" if eq_idx == 0 else "coral"
                hatch = None
                edgec = "black"
                alpha = 0.55 + 0.45 * float(p)
            else:
                color = "lightgray"
                hatch = "//"
                edgec = "gray"
                alpha = 0.35
            ax.barh(i, m, color=color, alpha=alpha,
                    edgecolor=edgec, linewidth=0.6, hatch=hatch)
            ax.errorbar(m, i, xerr=[[lo], [hi]],
                        fmt="none", ecolor="black", capsize=3, alpha=0.7)
            # P_incl annotation right of bar
            ax.text(max(m + hi, 0) + 0.02 * max(1.0, abs(m + hi)), i,
                    f"P={p:.2f}", va="center", fontsize=8, color=edgec)

        ax.set_yticks(ypos)
        ax.set_yticklabels(labs, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_xlabel("Coefficient median (with IQR)")
        ax.set_title(f"{name}  (filled = P_incl >= {inclusion_threshold:.2f})")

    plt.suptitle(f"{disease_name.title()} - Ensemble Coefficients (median +/- IQR)",
                 fontsize=14)
    plt.tight_layout()

    if save:
        filepath = os.path.join(output_dir, f"{disease_name}_coefficients_ensemble.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.close()


def plot_inclusion_heatmap(inclusion_prob, labels, disease_name,
                           inclusion_threshold=0.6, save=True,
                           output_dir=None):
    """
    Heatmap (2 columns x p rows) of per-term inclusion probabilities.

    Colors converge around the threshold so it's easy to read off which
    terms cleared 0.6 in either equation.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(labels) + 1)))
    im = ax.imshow(inclusion_prob, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["S equation", "I equation"], rotation=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(2):
            v = inclusion_prob[i, j]
            color = "white" if v < 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cb.set_label(f"Inclusion probability (threshold = {inclusion_threshold:.2f})")
    ax.set_title(f"{disease_name.title()} - Term Inclusion Probabilities")

    plt.tight_layout()
    if save:
        filepath = os.path.join(output_dir, f"{disease_name}_inclusion_heatmap.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.close()


def plot_regime_shift(regime_results, disease_name, save=True, output_dir=None):
    """
    Two-panel figure: I(t) trajectories and PSDs under each perturbation
    factor. Reproduces the qualitative content of paper Figs. 8 & 9 for
    a single perturbation sweep.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    factors = regime_results["factors"]
    I_sims = regime_results["I_sims"]
    freqs = regime_results["freqs"]
    psds = regime_results["psds"]
    periods = regime_results["dominant_period_years"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(factors) - 1)) for i in range(len(factors))]

    # --- Top: I(t) trajectories ----------------------------------------
    for f, I_sim, period, c in zip(factors, I_sims, periods, colors):
        years = np.arange(len(I_sim)) / 52.0
        period_str = f"{period:.2f}yr" if np.isfinite(period) else "n/a"
        ax1.plot(years, I_sim, color=c, alpha=0.85, linewidth=1.0,
                 label=f"factor={f:.2f}, period={period_str}")
    ax1.set_xlabel("Years from start")
    ax1.set_ylabel("Infectious prevalence I(t)")
    ax1.set_title(f"{disease_name.title()} - I(t) under S-equation perturbation")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Bottom: PSDs --------------------------------------------------
    for f, fr, ps, c in zip(factors, freqs, psds, colors):
        if len(fr) == 0:
            continue
        mask = (fr > 0) & (fr <= 3.0)
        if not np.any(mask):
            continue
        ax2.semilogy(fr[mask], np.maximum(ps[mask], 1e-30), color=c,
                     alpha=0.85, linewidth=1.0, label=f"factor={f:.2f}")
    ax2.set_xlabel("Frequency (cycles/year)")
    ax2.set_ylabel("Power spectral density")
    ax2.set_title("Power spectra of simulated I(t)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)

    detected = regime_results.get("regime_shift_detected", False)
    baseline = regime_results.get("baseline_period_years", float("nan"))
    suptitle = (f"Regime-shift test  "
                f"(baseline period {baseline:.2f}yr, "
                f"shift detected: {'YES' if detected else 'no'})")
    fig.suptitle(suptitle, fontsize=12, y=1.01)

    plt.tight_layout()
    if save:
        filepath = os.path.join(output_dir, f"{disease_name}_regime_shift.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.close()


def plot_method_comparison(records, disease_name, save=True, output_dir=None):
    """
    Side-by-side bar chart of {paper, baseline, ensemble} coefficient
    values for the headline I-equation terms.

    `records` is a list of dicts produced by results_io.result_to_record;
    we extract the rows for `disease_name` and read the columns
    paper_I_eq_*, I_eq_* (for both methods), and iqr_I_eq_* (ensemble
    only) to draw error bars.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    headline_terms = ["SI", "β(t)·SI", "S²", "I²", "β(t)·I²"]
    rows = [r for r in records if r["disease"] == disease_name]
    if not rows:
        return

    by_method = {r["method"]: r for r in rows}

    # Paper values can be read from any row via paper_I_eq_*
    if rows:
        paper_vals = [rows[0].get(f"paper_I_eq_{t}", float("nan"))
                      for t in headline_terms]
    else:
        paper_vals = [float("nan")] * len(headline_terms)
    base = by_method.get("baseline_sindy")
    ens = by_method.get("ensemble_sindy")

    base_vals = ([base.get(f"I_eq_{t}", float("nan")) for t in headline_terms]
                 if base else [float("nan")] * len(headline_terms))
    ens_vals = ([ens.get(f"I_eq_{t}", float("nan")) for t in headline_terms]
                if ens else [float("nan")] * len(headline_terms))
    ens_iqr = ([ens.get(f"iqr_I_eq_{t}", 0.0) for t in headline_terms]
               if ens else [0.0] * len(headline_terms))
    ens_p = ([ens.get(f"incl_prob_I_eq_{t}", float("nan"))
              for t in headline_terms] if ens else [float("nan")] * len(headline_terms))

    x = np.arange(len(headline_terms))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, paper_vals, w, label="Paper", color="steelblue", alpha=0.85)
    ax.bar(x, base_vals, w, label="Baseline reproduction",
           color="orange", alpha=0.85)
    ax.bar(x + w, ens_vals, w, yerr=np.array(ens_iqr) / 2,
           label="Ensemble (median +/- IQR/2)", color="seagreen", alpha=0.85,
           ecolor="black", capsize=3)

    # Annotate ensemble bars with inclusion probability
    for xi, v, p in zip(x + w, ens_vals, ens_p):
        if np.isfinite(p):
            ax.text(xi, (v if np.isfinite(v) else 0)
                    + 0.04 * max(abs(v) if np.isfinite(v) else 1.0, 1.0),
                    f"P={p:.2f}", ha="center", fontsize=8, color="darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels(headline_terms)
    ax.set_ylabel("Coefficient in I equation")
    ax.set_title(f"{disease_name.title()} - Paper vs Baseline vs Ensemble (I equation)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        filepath = os.path.join(output_dir, f"{disease_name}_method_comparison.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    plt.close()


def plot_psd_comparison(freq_data, psd_data, freq_model, psd_model,
                        disease_name, save=True):
    """
    Compare power spectral densities of data vs model.
    
    Parameters
    ----------
    freq_data, psd_data : arrays — PSD of empirical data
    freq_model, psd_model : arrays — PSD of model output
    disease_name : str
    save : bool
    """
    ensure_output_dir()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.semilogy(freq_data, psd_data, "b-", alpha=0.7, label="Data")
    ax.semilogy(freq_model, psd_model, "r--", alpha=0.8, label="SINDy model")
    ax.set_xlabel("Frequency (cycles/year)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"{disease_name.title()} — Power Spectral Density Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_psd.png")
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"  Saved: {filepath}")
    
    plt.close()
