# SINDy for Infectious Disease Data — Python Implementation

Complete Python reimplementation of the paper:
**"Algorithmic Discovery of Dynamic Models from Infectious Disease Data"**
by Horrocks & Bauch (2020), Scientific Reports.

## Project Structure

```
sindy_project/
├── README.md
├── requirements.txt
├── run_all.py                  # Entry point — runs entire pipeline
├── data/
│   ├── generate_data.py        # Generates synthetic + realistic disease data
│   ├── measles.csv             # Generated measles case data
│   ├── chickenpox.csv          # Generated chickenpox case data
│   └── rubella.csv             # Generated rubella case data
├── src/
│   ├── __init__.py
│   ├── config.py               # All parameters and constants
│   ├── preprocessing.py        # Smoothing, incidence→prevalence, susceptible reconstruction
│   ├── function_library.py     # Builds Θ(X) matrix — poolData equivalent
│   ├── sindy_core.py           # sparsifyDynamics — the core algorithm
│   ├── model_selection.py      # AIC computation, grid search over S₀, λ, φ
│   ├── simulation.py           # Simulate discovered model forward
│   ├── psd_analysis.py         # Power spectral density approach (rubella fix)
│   └── visualization.py        # All plotting functions
├── outputs/                    # Generated plots and results
└── notebooks/
    └── walkthrough.py          # Step-by-step demo script
```

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

## What This Does

1. Generates realistic synthetic disease data (measles, chickenpox, rubella patterns)
2. Preprocesses: smoothing → incidence-to-prevalence → susceptible reconstruction
3. Builds the function library Θ(X) with polynomial + seasonal terms
4. Runs SINDy with grid search over (S₀, λ, φ)
5. Selects best model via AIC
6. Simulates discovered model and compares to data
7. Generates all plots and coefficient tables
