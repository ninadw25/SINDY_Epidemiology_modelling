"""
psd_analysis.py — Power Spectral Density approach for model selection.

When fitting raw time series fails (rubella case), we can instead fit
the FREQUENCY CONTENT of the data. This selects models that reproduce
the correct cycle lengths even if exact outbreak timing differs.

Steps (from the paper):
    1. Smooth infection time series (moving average, window 23)
    2. Linear detrending
    3. Taper 20% with split cosine bell
    4. Compute periodogram
    5. AIC from PSD residuals instead of time-series residuals
"""

import numpy as np
from scipy.signal import periodogram, windows


def compute_psd(time_series, fs=52.0):
    """
    Compute the power spectral density of a time series.
    
    Following the paper's methodology:
    1. Moving average smoothing (window 23)
    2. Linear detrending
    3. Split cosine bell tapering (20%)
    4. Periodogram computation
    
    Parameters
    ----------
    time_series : array — the prevalence time series
    fs : float — sampling frequency (52 = weekly data → frequency in cycles/year)
    
    Returns
    -------
    frequencies : array — frequency axis (cycles per year)
    psd : array — power spectral density values
    """
    data = time_series.copy()
    n = len(data)
    
    # Step 1: Moving average smoothing (window 23)
    kernel = np.ones(min(23, n)) / min(23, n)
    data = np.convolve(data, kernel, mode="same")
    
    # Step 2: Linear detrending
    t = np.arange(n)
    slope, intercept = np.polyfit(t, data, 1)
    data = data - (slope * t + intercept)
    
    # Step 3: Split cosine bell tapering (20%)
    taper_frac = 0.2
    taper_n = int(n * taper_frac / 2)
    taper = np.ones(n)
    if taper_n > 0:
        bell = 0.5 * (1 - np.cos(np.pi * np.arange(taper_n) / taper_n))
        taper[:taper_n] = bell
        taper[-taper_n:] = bell[::-1]
    data = data * taper
    
    # Step 4: Periodogram
    frequencies, psd = periodogram(data, fs=fs)
    
    return frequencies, psd


def compute_aic_psd(psd_data, psd_model, n_params):
    """
    Compute AIC based on PSD residuals instead of time-series residuals.
    
    AIC_PSD = 2k + n × ln(RSS_PSD / n)
    where RSS_PSD = Σ(PSD_data − PSD_model)²
    
    Parameters
    ----------
    psd_data : array — PSD of empirical data
    psd_model : array — PSD of model output
    n_params : int — number of free parameters
    
    Returns
    -------
    aic : float — AIC score (lower = better)
    """
    n = min(len(psd_data), len(psd_model))
    residuals = psd_data[:n] - psd_model[:n]
    rss = np.sum(residuals ** 2)
    
    if rss < 1e-30:
        rss = 1e-30
    
    aic = 2 * n_params + n * np.log(rss / n)
    return aic
