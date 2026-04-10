"""
preprocessing.py — Data preprocessing pipeline.

Three transformations to go from raw case counts to SINDy input:

1. Smoothing (Savitzky-Golay filter)
   - Reduces noise while preserving peak shapes
   - Paper uses order 3, window 19

2. Incidence to Prevalence conversion (Equation 20)
   - P_t = C_t × p × D_i / (⟨C_t⟩ × L)
   - Converts "new cases this week" to "total currently infected"

3. Susceptible Reconstruction (Finkenstädt-Grenfell method)
   - S_{t+1} = S_t − α·C_{t,t+1} + B_{t,t+1}
   - Reconstructs the hidden susceptible pool from births and cases
"""

import numpy as np
from scipy.signal import savgol_filter
from src.config import SMOOTHING


def smooth_cases(cases, window_length=None, polyorder=None):
    """
    Apply Savitzky-Golay filter to smooth noisy case data.
    
    The paper uses this to reduce the risk of SINDy overfitting noise.
    It fits a local polynomial to each window of data points, which
    smooths out noise while preserving the shape of peaks and valleys.
    
    Parameters
    ----------
    cases : array — raw weekly case counts
    window_length : int — must be odd (default: 19 from config)
    polyorder : int — polynomial order (default: 3 from config)
    
    Returns
    -------
    smoothed : array — smoothed case counts
    
    MATLAB equivalent: sgolayfilt(data, 3, 19)
    """
    if window_length is None:
        window_length = SMOOTHING["window_length"]
    if polyorder is None:
        polyorder = SMOOTHING["polyorder"]
    
    # Ensure window length doesn't exceed data length
    if window_length > len(cases):
        window_length = len(cases) if len(cases) % 2 == 1 else len(cases) - 1
    
    smoothed = savgol_filter(cases, window_length, polyorder)
    
    # Case counts can't be negative
    smoothed = np.maximum(smoothed, 0)
    
    return smoothed


def incidence_to_prevalence(cases_smoothed, D_i, p, L, population):
    """
    Convert incidence (new cases/week) to prevalence (fraction currently infected).
    
    Equation 20 from the paper:
        P_t = C_t × p × D_i / (⟨C_t⟩ × L)
    
    Why this is needed:
    - The data gives INCIDENCE: how many people got sick this week
    - The SIR model needs PREVALENCE: how many people ARE sick right now
    - A disease lasting 2 weeks means this week's sick include both this 
      week's new cases AND last week's cases who haven't recovered yet
    
    Parameters
    ----------
    cases_smoothed : array — smoothed weekly incidence counts
    D_i : float — duration of infectiousness in weeks (typically 2)
    p : float — lifetime infection probability (0.957)
    L : float — mean lifespan in weeks (65 years × 52)
    population : int — total population size
    
    Returns
    -------
    I_t : array — infection prevalence as proportion of population
    """
    mean_cases = np.mean(cases_smoothed)
    
    # Avoid division by zero
    if mean_cases < 1e-10:
        return np.zeros_like(cases_smoothed)
    
    # Equation 20: P_t = C_t × p × D_i / (⟨C_t⟩ × L)
    P_t = cases_smoothed * p * D_i / (mean_cases * L)
    
    # Ensure non-negative
    P_t = np.maximum(P_t, 0)
    
    return P_t


def reconstruct_susceptible(cases_smoothed, births, population, S0_fraction=0.10):
    """
    Reconstruct the susceptible time series using the Finkenstädt-Grenfell method.
    
    Since nobody ever measured how many people were susceptible to measles
    in 1955, we must INFER it from data we DO have: cases and births.
    
    Basic logic:
        S(next week) = S(this week) + births(this week) − infections(this week)
    
    But not all cases are reported. The reporting rate α is estimated by
    linear regression of cumulative births vs cumulative cases:
        Y_t = ᾱ × X_t + (Z_t − Z₀)
    
    where X_t = cumulative cases, Y_t = cumulative births.
    The slope gives ᾱ, the residuals give susceptible deviations.
    
    Parameters
    ----------
    cases_smoothed : array — smoothed weekly incidence counts
    births : array — weekly birth counts
    population : int — total population
    S0_fraction : float — initial susceptible fraction (parameter to search over)
    
    Returns
    -------
    S_t : array — susceptible proportion of population over time
    alpha : float — estimated reporting rate
    """
    n = len(cases_smoothed)
    
    # Cumulative sums (Equation 15 notation)
    X_t = np.cumsum(cases_smoothed)   # cumulative cases
    Y_t = np.cumsum(births)           # cumulative births
    
    # --- Global regression to estimate reporting rate α ---
    # Y_t = ᾱ × X_t + constant (Equation 17)
    # Slope of Y vs X gives the average reporting rate
    
    if np.std(X_t) < 1e-10:
        # No variation in cases — can't estimate
        alpha = 0.5
    else:
        # Simple linear regression: slope = cov(X,Y) / var(X)
        alpha = np.cov(X_t, Y_t)[0, 1] / np.var(X_t)
        alpha = np.clip(alpha, 0.01, 1.0)  # reporting rate must be in (0, 1]
    
    # --- Reconstruct susceptible deviations Z_t ---
    # Z_t = Z₀ − ᾱ·X_t + Y_t  (Equation 16, simplified)
    # Then S_t = S̄ + Z_t
    
    Z_t = -alpha * X_t + Y_t
    Z_deviations = Z_t - Z_t[0]  # deviations from initial
    
    # Convert to susceptible proportion
    S_mean = S0_fraction  # the mean susceptible level
    S_t = S_mean + Z_deviations / population
    
    # Clamp to valid range
    S_t = np.clip(S_t, 0.01, 0.99)
    
    return S_t, alpha


def preprocess_disease(cases, births, population, D_i, p, L, S0_fraction=0.10):
    """
    Full preprocessing pipeline: raw cases → (I_t, S_t) ready for SINDy.
    
    This is the complete data pipeline from Figure 2 of the visual guide:
    
    Raw cases → Smooth → Incidence→Prevalence → I(t)
                     ↘ → Susceptible Reconstruction → S(t)
    
    Parameters
    ----------
    cases : array — raw weekly case counts
    births : array — weekly birth counts
    population : int — total population
    D_i : float — infection duration (weeks)
    p : float — lifetime infection probability
    L : float — lifespan (weeks)
    S0_fraction : float — initial susceptible fraction
    
    Returns
    -------
    S_t : array — susceptible proportion time series
    I_t : array — infectious prevalence proportion time series
    cases_smooth : array — smoothed case counts
    alpha : float — estimated reporting rate
    """
    # Step 1: Smooth
    cases_smooth = smooth_cases(cases)
    
    # Step 2: Incidence → Prevalence
    I_t = incidence_to_prevalence(cases_smooth, D_i, p, L, population)
    
    # Step 3: Reconstruct susceptibles
    S_t, alpha = reconstruct_susceptible(cases_smooth, births, population, S0_fraction)
    
    return S_t, I_t, cases_smooth, alpha
