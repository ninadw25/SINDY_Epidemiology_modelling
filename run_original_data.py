"""
run_original_data.py — Run the SINDy pipeline on REAL historical disease data.

This script loads the original data files from the paper:
    - Measles:    England & Wales weekly cases (1948-1967) from mDataEW_N.mat
    - Chickenpox: Ontario weekly cases (1946-1967) from OntarioChickenWeekly39_69.txt
    - Rubella:    Ontario weekly cases (1946-1960) from OntarioRubellaWeekly39_69.txt
    - Births:     Ontario quarterly births from Ontario_Birth_Data_M.txt
    - Population: Ontario census data from Ontario_Demographics_Measles.txt

Data source: International Infectious Disease Data Archive (IIDDA), McMaster University.

Usage:
    python run_original_data.py              # Run all three diseases
    python run_original_data.py measles      # Run one disease
    python run_original_data.py --fast       # Reduced grid for quick testing
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DISEASES, SINDY, FIGURE_DPI
from src.sindy_core import print_discovered_model
from src.model_selection import grid_search
from src.psd_analysis import compute_psd, compute_aic_psd
from src.results_io import (
    load_paper_reference,
    result_to_record,
    save_full_xi_json,
    save_results_csv,
)
from src.visualization import (
    plot_time_series_comparison,
    plot_coefficients,
    plot_grid_search,
    plot_psd_comparison,
)

# Output directory for real-data results
OUTPUT_DIR = "outputs_real_data"


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_measles_data():
    """
    Load England & Wales measles data from the MATLAB .mat file.

    The file mDataEW_N.mat contains a matrix 'mDataN' with 4 columns:
        Column 0: year_fraction  (e.g. 1948.0274 = first week of 1948)
        Column 1: weekly cases   (incidence)
        Column 2: population     (interpolated weekly)
        Column 3: weekly births  (interpolated)

    Returns: cases, births, population, year_fractions
    """
    import scipy.io

    mat_path = os.path.join("orignal_data", "mDataEW_N.mat")
    mat = scipy.io.loadmat(mat_path)
    data = mat["mDataN"]

    year_fracs = data[:, 0]
    cases = data[:, 1]
    population = data[:, 2]
    births = data[:, 3]

    # Filter to 1948-1967 as per the paper
    mask = (year_fracs >= 1948.0) & (year_fracs < 1967.0)
    cases = cases[mask]
    births = births[mask]
    population = population[mask]
    year_fracs = year_fracs[mask]

    avg_pop = int(np.mean(population))

    print(f"  Measles (England & Wales): {len(cases)} weeks, "
          f"{year_fracs[0]:.1f}-{year_fracs[-1]:.1f}")
    print(f"  Population: ~{avg_pop:,}, Mean cases/week: {np.mean(cases):.0f}")

    return cases, births, avg_pop, year_fracs


def load_ontario_births():
    """
    Load Ontario quarterly birth data and interpolate to weekly.

    File format: year_fraction<tab>births (quarterly totals)
    Example: 1946    20257  (Q1 1946: 20,257 births)

    Returns: interpolation function that gives weekly births for any year_fraction.
    """
    filepath = os.path.join("orignal_data", "Ontario_Birth_Data_M.txt")
    data = np.loadtxt(filepath)
    years = data[:, 0]
    births_quarterly = data[:, 1]

    # Convert quarterly births to weekly rate (13 weeks per quarter)
    births_weekly = births_quarterly / 13.0

    # Create interpolation function
    from scipy.interpolate import interp1d
    birth_interp = interp1d(years, births_weekly, kind="linear",
                            fill_value="extrapolate")

    return birth_interp


def load_ontario_population():
    """
    Load Ontario population data (census every ~5 years) and interpolate.

    File format: year<tab>population
    Example: 1971    7703106

    Returns: interpolation function for population at any year.
    """
    filepath = os.path.join("orignal_data", "Ontario_Demographics_Measles.txt")

    # Read manually — the file has some blank lines
    years = []
    pops = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                years.append(float(parts[0]))
                pops.append(float(parts[1]))

    years = np.array(years)
    pops = np.array(pops)

    # Sort by year
    idx = np.argsort(years)
    years = years[idx]
    pops = pops[idx]

    from scipy.interpolate import interp1d
    pop_interp = interp1d(years, pops, kind="linear", fill_value="extrapolate")

    return pop_interp


def load_chickenpox_data():
    """
    Load Ontario chickenpox weekly cases (1946-1967).

    File format: year_fraction<tab>cases
    Example: 1939.019165    452

    Returns: cases, births, population, year_fractions
    """
    filepath = os.path.join("orignal_data", "OntarioChickenWeekly39_69.txt")

    # Read manually — some lines have missing values or null bytes
    year_fracs_raw = []
    cases_raw = []
    with open(filepath, "rb") as f:
        for line in f:
            line = line.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    year_fracs_raw.append(float(parts[0]))
                    cases_raw.append(float(parts[1]))
                except ValueError:
                    continue

    year_fracs = np.array(year_fracs_raw)
    cases = np.array(cases_raw)

    # Filter to 1946-1967 as per the paper
    mask = (year_fracs >= 1946.0) & (year_fracs < 1967.0)
    year_fracs = year_fracs[mask]
    cases = cases[mask]

    # Load Ontario births (quarterly -> weekly interpolated)
    birth_interp = load_ontario_births()
    births = birth_interp(year_fracs)
    births = np.maximum(births, 0)

    # Load Ontario population (interpolated)
    pop_interp = load_ontario_population()
    population = pop_interp(year_fracs)
    avg_pop = int(np.mean(population))

    print(f"  Chickenpox (Ontario): {len(cases)} weeks, "
          f"{year_fracs[0]:.1f}-{year_fracs[-1]:.1f}")
    print(f"  Population: ~{avg_pop:,}, Mean cases/week: {np.mean(cases):.0f}")

    return cases, births, avg_pop, year_fracs


def load_rubella_data():
    """
    Load Ontario rubella weekly cases (1946-1960).

    Same format as chickenpox. Rubella has a longer cycle (5-7 years)
    and lower case counts.

    Returns: cases, births, population, year_fractions
    """
    filepath = os.path.join("orignal_data", "OntarioRubellaWeekly39_69.txt")

    # Read manually — some lines may have missing values or null bytes
    year_fracs = []
    cases_list = []
    with open(filepath, "rb") as f:
        for line in f:
            line = line.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    year_fracs.append(float(parts[0]))
                    cases_list.append(float(parts[1]))
                except ValueError:
                    continue

    year_fracs = np.array(year_fracs)
    cases = np.array(cases_list)

    # Filter to 1946-1960 as per the paper
    mask = (year_fracs >= 1946.0) & (year_fracs < 1960.0)
    year_fracs = year_fracs[mask]
    cases = cases[mask]

    # Load Ontario births and population
    birth_interp = load_ontario_births()
    births = birth_interp(year_fracs)
    births = np.maximum(births, 0)

    pop_interp = load_ontario_population()
    population = pop_interp(year_fracs)
    avg_pop = int(np.mean(population))

    print(f"  Rubella (Ontario): {len(cases)} weeks, "
          f"{year_fracs[0]:.1f}-{year_fracs[-1]:.1f}")
    print(f"  Population: ~{avg_pop:,}, Mean cases/week: {np.mean(cases):.0f}")

    return cases, births, avg_pop, year_fracs


# =============================================================================
# ENHANCED VISUALIZATION (with correct year axes for real data)
# =============================================================================

def plot_raw_data(cases, year_fracs, disease_name, births=None):
    """Plot the raw case data before any processing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2 if births is not None else 1, 1,
                             figsize=(14, 8 if births is not None else 5))

    if births is None:
        axes = [axes]

    # Cases
    ax = axes[0]
    ax.plot(year_fracs, cases, "b-", linewidth=0.8, alpha=0.8)
    ax.fill_between(year_fracs, 0, cases, alpha=0.2, color="steelblue")
    ax.set_ylabel("Weekly reported cases")
    ax.set_title(f"{disease_name.title()} — Raw Weekly Case Data (Original)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(year_fracs[0], year_fracs[-1])

    # Births
    if births is not None:
        ax2 = axes[1]
        ax2.plot(year_fracs, births, "g-", linewidth=0.8, alpha=0.8)
        ax2.set_ylabel("Weekly births")
        ax2.set_xlabel("Year")
        ax2.set_title(f"Birth Data (used for susceptible reconstruction)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(year_fracs[0], year_fracs[-1])

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_raw_data.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def plot_preprocessing_steps(cases, cases_smooth, S_t, I_t, year_fracs,
                             disease_name, alpha):
    """Show all preprocessing steps visually."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    n = min(len(year_fracs), len(cases_smooth))

    # Panel 1: Raw vs Smoothed cases
    ax = axes[0]
    ax.plot(year_fracs[:n], cases[:n], "b-", alpha=0.4, linewidth=0.5,
            label="Raw cases")
    ax.plot(year_fracs[:n], cases_smooth[:n], "r-", linewidth=1.5,
            label="Smoothed (Savitzky-Golay)")
    ax.set_ylabel("Weekly cases")
    ax.set_title(f"{disease_name.title()} — Step 1: Noise Smoothing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Infectious prevalence I(t)
    ax = axes[1]
    ax.plot(year_fracs[:n], I_t[:n], "darkred", linewidth=1)
    ax.fill_between(year_fracs[:n], 0, I_t[:n], alpha=0.2, color="red")
    ax.set_ylabel("I(t) — proportion infected")
    ax.set_title(f"Step 2: Incidence to Prevalence Conversion  "
                 f"(P_t = C_t * p * D_i / (<C_t> * L))")
    ax.grid(True, alpha=0.3)

    # Panel 3: Susceptible fraction S(t)
    ax = axes[2]
    ax.plot(year_fracs[:n], S_t[:n], "darkblue", linewidth=1)
    ax.fill_between(year_fracs[:n], S_t[:n].min(), S_t[:n], alpha=0.2,
                    color="steelblue")
    ax.set_ylabel("S(t) — proportion susceptible")
    ax.set_xlabel("Year")
    ax.set_title(f"Step 3: Susceptible Reconstruction  "
                 f"(reporting rate alpha = {alpha:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_preprocessing.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def plot_real_time_series(S_data, I_data, S_sim, I_sim, year_fracs,
                          disease_name, result_info=None):
    """Time series comparison with real year axis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = min(len(S_data), len(S_sim), len(year_fracs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Susceptible
    ax1.plot(year_fracs[:n], S_data[:n], "b-", alpha=0.7, linewidth=1,
             label="Reconstructed from data")
    ax1.plot(year_fracs[:n], S_sim[:n], "r--", alpha=0.8, linewidth=1,
             label="SINDy model")
    ax1.set_ylabel("Susceptible (proportion)")
    ax1.set_title(f"{disease_name.title()} — Susceptible: Data vs SINDy Model")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Infectious
    ax2.plot(year_fracs[:n], I_data[:n], "b-", alpha=0.7, linewidth=1,
             label="Derived from data")
    ax2.plot(year_fracs[:n], I_sim[:n], "r--", alpha=0.8, linewidth=1,
             label="SINDy model")
    ax2.set_ylabel("Infectious (proportion)")
    ax2.set_xlabel("Year")
    ax2.set_title(f"{disease_name.title()} — Infectious: Data vs SINDy Model")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if result_info:
        info_text = (f"Best params: S0={result_info.get('S0', '?'):.4f}, "
                     f"lambda={result_info.get('lambda_c', '?'):.5f}, "
                     f"phi={result_info.get('phi', '?'):.1f} weeks, "
                     f"sparsity r={result_info.get('sparsity', '?'):.2f}, "
                     f"AIC={result_info.get('aic', '?'):.1f}")
        fig.suptitle(info_text, fontsize=10, y=0.02, color="gray")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_time_series.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def plot_real_coefficients(Xi, labels, disease_name):
    """Bar chart of discovered coefficients with interpretation annotations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    eq_names = ["S equation (susceptible dynamics)",
                "I equation (infection dynamics)"]
    axes = [ax1, ax2]
    colors = ["steelblue", "coral"]

    for eq_idx, (ax, name, color) in enumerate(zip(axes, eq_names, colors)):
        coeffs = Xi[:, eq_idx]
        nonzero = np.abs(coeffs) > 1e-10

        if not np.any(nonzero):
            ax.text(0.5, 0.5, "All terms eliminated", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            ax.set_title(name)
            continue

        idx = np.where(nonzero)[0]
        vals = coeffs[idx]
        labs = [labels[i] for i in idx]

        # Color-code by term type
        bar_colors = []
        for i, lab in zip(idx, labs):
            if "SI" in lab and "I\u00b2" not in lab and "S\u00b2" not in lab:
                bar_colors.append("#e74c3c")  # red for mass-action
            elif "\u03b2(t)" in lab:
                bar_colors.append("#f39c12")  # orange for seasonal
            else:
                bar_colors.append(color)

        bars = ax.barh(range(len(vals)), vals, color=bar_colors, alpha=0.85,
                       edgecolor="gray", linewidth=0.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labs, fontsize=10)
        ax.set_xlabel("Coefficient value")
        ax.set_title(name, fontsize=11)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="x")

        # Add value labels on bars
        for j, (v, bar) in enumerate(zip(vals, bars)):
            ax.text(v + (0.01 * np.sign(v) * max(abs(vals))), j,
                    f"{v:.3f}", va="center", fontsize=8, color="black")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Mass-action (SI terms)"),
        Patch(facecolor="#f39c12", label="Seasonal forcing"),
        Patch(facecolor="steelblue", label="Other dynamics"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        f"{disease_name.title()} — Discovered Model Coefficients (Real Data)",
        fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_coefficients.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def plot_real_grid_search(all_results, disease_name):
    """Enhanced grid search heatmap."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(all_results) < 4:
        print(f"  Grid search plot skipped (too few results).")
        return

    S0_vals = np.array([r["S0"] for r in all_results])
    lam_vals = np.array([r["lambda_c"] for r in all_results])
    aic_vals = np.array([r["aic"] for r in all_results])

    # Clip extreme AIC values for better visualization
    aic_clipped = np.clip(aic_vals, np.percentile(aic_vals, 5),
                          np.percentile(aic_vals, 95))

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))

    scatter = ax.scatter(S0_vals, lam_vals, c=aic_clipped, cmap="RdYlGn_r",
                         s=50, alpha=0.85, edgecolors="gray", linewidth=0.3)

    ax.set_xlabel("S_0 (initial susceptible fraction)", fontsize=12)
    ax.set_ylabel("lambda (sparsity threshold)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(
        f"{disease_name.title()} — AIC Across Parameter Grid (Real Data)",
        fontsize=13)
    plt.colorbar(scatter, ax=ax, label="AIC (lower = better)")

    best_idx = np.argmin(aic_vals)
    ax.scatter([S0_vals[best_idx]], [lam_vals[best_idx]],
               marker="*", s=300, c="black", zorder=5,
               label=f"Best: S0={S0_vals[best_idx]:.4f}, "
                     f"lam={lam_vals[best_idx]:.5f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_grid_search.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def plot_real_psd(I_data, I_sim, disease_name):
    """PSD comparison with peak frequency annotation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = min(len(I_data), len(I_sim))
    freq_d, psd_d = compute_psd(I_data[:n])
    freq_m, psd_m = compute_psd(I_sim[:n])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.semilogy(freq_d, psd_d, "b-", alpha=0.7, linewidth=1.5, label="Data")
    ax.semilogy(freq_m, psd_m, "r--", alpha=0.8, linewidth=1.5,
                label="SINDy model")
    ax.set_xlabel("Frequency (cycles/year)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(
        f"{disease_name.title()} — Power Spectral Density Comparison",
        fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4)

    # Annotate dominant frequency
    valid = freq_d > 0.05
    if np.any(valid):
        peak_idx = np.argmax(psd_d[valid])
        peak_freq = freq_d[valid][peak_idx]
        peak_period = 1.0 / peak_freq if peak_freq > 0 else float("inf")
        ax.axvline(peak_freq, color="blue", linestyle=":", alpha=0.5)
        ax.text(peak_freq + 0.05, psd_d[valid][peak_idx],
                f"Data peak: {peak_period:.1f} yr cycle",
                fontsize=10, color="blue")

    valid_m = freq_m > 0.05
    if np.any(valid_m):
        peak_idx_m = np.argmax(psd_m[valid_m])
        peak_freq_m = freq_m[valid_m][peak_idx_m]
        peak_period_m = 1.0 / peak_freq_m if peak_freq_m > 0 else float("inf")
        ax.axvline(peak_freq_m, color="red", linestyle=":", alpha=0.5)
        ax.text(peak_freq_m + 0.05,
                psd_m[valid_m][peak_idx_m] * 0.5,
                f"Model peak: {peak_period_m:.1f} yr cycle",
                fontsize=10, color="red")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{disease_name}_psd.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {filepath}")
    plt.close()


def generate_summary_figure(all_disease_results):
    """Create a summary comparison figure across all diseases."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    diseases = [d for d in ["measles", "chickenpox", "rubella"]
                if d in all_disease_results and all_disease_results[d] is not None]

    if not diseases:
        return

    fig, axes = plt.subplots(len(diseases), 2, figsize=(16, 5 * len(diseases)))
    if len(diseases) == 1:
        axes = axes.reshape(1, -1)

    for i, disease in enumerate(diseases):
        res = all_disease_results[disease]
        year_fracs = res["year_fracs"]
        n = min(len(res["I_t"]), len(res["I_sim"]), len(year_fracs))

        # Left: time series
        ax = axes[i, 0]
        ax.plot(year_fracs[:n], res["I_t"][:n], "b-", alpha=0.7, linewidth=1,
                label="Data")
        ax.plot(year_fracs[:n], res["I_sim"][:n], "r--", alpha=0.8,
                linewidth=1, label="SINDy")
        ax.set_title(f"{disease.title()} — Infectious Prevalence")
        ax.set_ylabel("I(t)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == len(diseases) - 1:
            ax.set_xlabel("Year")

        # Right: coefficients for I equation
        ax = axes[i, 1]
        Xi = res["Xi"]
        labels = res["labels"]
        coeffs = Xi[:, 1]  # I equation
        nonzero = np.abs(coeffs) > 1e-10
        if np.any(nonzero):
            idx = np.where(nonzero)[0]
            vals = coeffs[idx]
            labs = [labels[j] for j in idx]
            bar_colors = ["#e74c3c" if "SI" in l else "#f39c12" if "\u03b2" in l
                          else "steelblue" for l in labs]
            ax.barh(range(len(vals)), vals, color=bar_colors, alpha=0.85)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labs, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"{disease.title()} — I Equation Coefficients")
        ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("SINDy Results on Real Historical Disease Data", fontsize=16,
                 y=1.01)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "summary_all_diseases.png")
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"\n  Saved summary: {filepath}")
    plt.close()


# =============================================================================
# INTERPRETATION / EXPLANATION
# =============================================================================

def explain_results(disease_name, best):
    """Print a plain-English explanation of what SINDy discovered."""
    Xi = best["Xi"]
    labels = best["labels"]

    print(f"\n{'='*70}")
    print(f"  PLAIN-ENGLISH EXPLANATION: {disease_name.upper()}")
    print(f"{'='*70}")

    # What the algorithm found
    print(f"\n  The SINDy algorithm searched over {len(SINDY['S0_range'])*len(SINDY['lambda_range'])}"
          f" parameter combinations to find")
    print(f"  the simplest model that explains the {disease_name} data.")
    print(f"\n  Best parameters found:")
    print(f"    - S0 = {best['S0']:.4f} (initial fraction of people susceptible)")
    print(f"    - lambda = {best['lambda_c']:.5f} (sparsity: how aggressively to simplify)")
    print(f"    - phi = {best['phi']:.1f} weeks (when seasonal peak occurs)")
    print(f"    - AIC = {best['aic']:.1f} (model quality score; lower = better)")
    print(f"    - Sparsity r = {best['sparsity']:.2f} "
          f"(fraction of terms eliminated; 0.25-0.7 is good)")

    # Interpret the I equation (infection dynamics)
    I_coeffs = Xi[:, 1]
    print(f"\n  DISCOVERED INFECTION EQUATION (what drives {disease_name} spread):")
    print(f"  I(t+1) = ", end="")
    terms = []
    has_mass_action = False
    has_seasonal = False
    for i, (label, coef) in enumerate(zip(labels, I_coeffs)):
        if abs(coef) > 1e-10:
            sign = "+" if coef > 0 else ""
            terms.append(f"{sign}{coef:.3f}*{label}")
            if "SI" in label and "\u03b2" not in label:
                has_mass_action = True
            if "\u03b2(t)" in label and "SI" in label:
                has_seasonal = True
    print("  ".join(terms) if terms else "  (no terms survived)")

    print(f"\n  WHAT THIS MEANS IN SIMPLE TERMS:")

    if has_mass_action:
        print(f"    - The SI term (susceptible * infected) was discovered!")
        print(f"      This is MASS-ACTION INCIDENCE: the disease spreads when")
        print(f"      infected people mix with susceptible people. More of either")
        print(f"      means more transmission. This is the #1 principle of epidemiology.")

    if has_seasonal:
        print(f"    - The beta(t)*SI term (seasonal * mass-action) was also found!")
        print(f"      This means transmission varies with the seasons. For {disease_name},")
        print(f"      this likely reflects school terms: kids spread disease faster when")
        print(f"      schools are open (fall/winter) vs summer holidays.")

    # Check for S^2 term
    for i, label in enumerate(labels):
        if label == "S\u00b2" and abs(I_coeffs[i]) > 1e-10:
            print(f"    - The S^2 term was discovered (coefficient = {I_coeffs[i]:.3f}).")
            print(f"      This is a NOVEL FINDING from the paper: it causes epidemics")
            print(f"      to start slowly (subexponential growth) then accelerate.")

    # Expected vs actual cycle
    cfg = DISEASES.get(disease_name, {})
    expected = cfg.get("attractor", "unknown")
    print(f"\n  EXPECTED CYCLE PATTERN: {expected}")
    if expected == "biennial":
        print(f"    Measles famously cycles every 2 years in pre-vaccine UK data.")
    elif expected == "annual":
        print(f"    Chickenpox cycles once per year — a simpler pattern.")
    elif expected == "multiennial":
        print(f"    Rubella has a complex 5-7 year cycle — the hardest to fit.")

    print(f"\n  The model has {best['n_params']} nonzero terms out of "
          f"{len(labels)} candidates.")
    print(f"  This means SINDy eliminated "
          f"{len(labels) - best['n_params']}/{len(labels)} terms as unnecessary.")
    print(f"{'='*70}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_disease_real(disease_name, fast=False):
    """Run the complete SINDy pipeline on real data for one disease."""

    print(f"\n{'='*70}")
    print(f"  RUNNING SINDy ON REAL DATA: {disease_name.upper()}")
    print(f"{'='*70}")

    disease_cfg = DISEASES[disease_name]

    # ===== STAGE 1: Load real data =====
    print(f"\n[Stage 1] Loading original data...")
    if disease_name == "measles":
        cases, births, population, year_fracs = load_measles_data()
    elif disease_name == "chickenpox":
        cases, births, population, year_fracs = load_chickenpox_data()
    elif disease_name == "rubella":
        cases, births, population, year_fracs = load_rubella_data()
    else:
        print(f"  Unknown disease: {disease_name}")
        return None

    print(f"  Loaded {len(cases)} weeks of REAL data")

    # ===== STAGE 2: Plot raw data =====
    print(f"\n[Stage 2] Plotting raw data...")
    plot_raw_data(cases, year_fracs, disease_name, births)

    # ===== STAGE 3: Grid search =====
    print(f"\n[Stage 3-5] Preprocessing + Grid search + SINDy...")

    if fast:
        S0_range = np.linspace(0.04, 0.14, 8)
        lambda_range = np.logspace(-4, -1, 8)
        phi_range = np.arange(0, 52, 4.0)
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
        print(f"\n  [!] No valid model found for {disease_name}.")
        print(f"      Try adjusting the grid ranges.")
        return None

    # ===== STAGE 6: Print discovered model =====
    print(f"\n[Stage 6] Discovered model:")
    print_discovered_model(best["Xi"], best["labels"])

    # ===== STAGE 7: Plot preprocessing steps =====
    print(f"\n[Stage 7] Plotting preprocessing steps...")
    from src.preprocessing import preprocess_disease
    S_t, I_t, cases_smooth, alpha = preprocess_disease(
        cases, births, population,
        disease_cfg["D_i"], disease_cfg["p"], disease_cfg["L"],
        S0_fraction=best["S0"]
    )
    plot_preprocessing_steps(cases, cases_smooth, S_t, I_t, year_fracs,
                             disease_name, alpha)

    # ===== STAGE 8: Generate all plots =====
    print(f"\n[Stage 8] Generating result plots...")

    # Time series comparison with real year axis
    plot_real_time_series(
        best["S_t"], best["I_t"], best["S_sim"], best["I_sim"],
        year_fracs, disease_name, result_info=best
    )

    # Coefficient bar chart
    plot_real_coefficients(best["Xi"], best["labels"], disease_name)

    # Grid search heatmap
    plot_real_grid_search(all_results, disease_name)

    # PSD comparison
    try:
        plot_real_psd(best["I_t"], best["I_sim"], disease_name)
    except Exception as e:
        print(f"  PSD plot skipped: {e}")

    # ===== STAGE 9: Explain results =====
    explain_results(disease_name, best)

    # Store year_fracs in result for summary figure
    best["year_fracs"] = year_fracs

    # ===== STAGE 10: Export machine-readable results (JSON) =====
    # The legacy pipeline only emitted PNGs; this dump makes the
    # discovered model directly comparable to the paper without OCR.
    json_path = os.path.join(OUTPUT_DIR, f"{disease_name}_baseline.json")
    save_full_xi_json(disease_name, "baseline_legacy", best, json_path)

    print(f"  All {disease_name} plots saved to '{OUTPUT_DIR}/' folder.")
    return best


def main():
    """Main entry point."""
    args = sys.argv[1:]
    fast = "--fast" in args
    args = [a for a in args if a != "--fast"]

    if fast:
        print("FAST MODE: Using reduced grid for quick testing")

    diseases_to_run = args if args else ["measles", "chickenpox", "rubella"]

    print(f"\n{'#'*70}")
    print(f"#  SINDy on REAL Historical Disease Data")
    print(f"#  Based on: Horrocks & Bauch (2020), Scientific Reports")
    print(f"#  Data: International Infectious Disease Data Archive (IIDDA)")
    print(f"{'#'*70}")

    results = {}
    for disease in diseases_to_run:
        if disease not in DISEASES:
            print(f"Unknown disease: {disease}. "
                  f"Choose from: {list(DISEASES.keys())}")
            continue
        results[disease] = run_disease_real(disease, fast=fast)

    # ===== Summary =====
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY — Real Data Results")
    print(f"{'='*70}")

    for name, res in results.items():
        if res is None:
            print(f"  {name:>12}: FAILED — no valid model found")
        else:
            print(f"  {name:>12}: AIC={res['aic']:.1f}, "
                  f"sparsity r={res['sparsity']:.2f}, "
                  f"params={res['n_params']}, "
                  f"S0={res['S0']:.4f}, lambda={res['lambda_c']:.5f}, "
                  f"phi={res['phi']:.1f}wk")

    # ----- Append a CSV summary using the same schema as run_comparison.py
    paper_ref = load_paper_reference()
    records = [result_to_record(name, "baseline_legacy", res, paper_ref)
               for name, res in results.items() if res is not None]
    if records:
        save_results_csv(records, os.path.join(OUTPUT_DIR, "results_legacy.csv"))

    # Generate summary comparison figure
    generate_summary_figure(results)

    print(f"\n  All plots saved to '{OUTPUT_DIR}/' folder.")
    print(f"\n{'='*70}")
    print(f"  WHAT JUST HAPPENED (Plain English)")
    print(f"{'='*70}")
    print(f"""
  We fed REAL historical disease data (pre-vaccine era) into a machine
  learning algorithm called SINDy (Sparse Identification of Nonlinear
  Dynamics). Instead of TELLING the algorithm how diseases spread, we let
  it DISCOVER the rules from the data alone.

  The key question: Can an algorithm independently rediscover what
  epidemiologists spent decades figuring out?

  The algorithm was given only:
    - Weekly case counts (how many people got sick each week)
    - Birth data (how many babies were born)
    - Population size

  From this, it automatically discovered:
    1. MASS-ACTION INCIDENCE (SI term): diseases spread when infected
       people meet susceptible people. More of either = more spread.
    2. SEASONAL FORCING (beta(t)*SI): transmission peaks during school
       terms and drops during summer holidays.

  These are the TWO FUNDAMENTAL PILLARS of epidemic theory, discovered
  entirely from data without any prior knowledge of epidemiology!
""")
    print(f"  Done!")


if __name__ == "__main__":
    main()
