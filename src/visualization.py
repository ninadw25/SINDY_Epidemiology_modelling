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
