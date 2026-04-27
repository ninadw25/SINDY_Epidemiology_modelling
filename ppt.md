# Algorithmic Discovery of Disease Models
## Reproducing Horrocks & Bauch (2020) with an Ensemble-SINDy Upgrade
*End-semester project — ML in Healthcare — April 2026*

---

## Slide 1 — Title

**Algorithmic Discovery of Disease Models**

Reproducing Horrocks & Bauch (2020) with an Ensemble-SINDy Upgrade

End-semester project — ML in Healthcare

---

## Slide 2 — The Big Question

[FIGURE 1]

- Traditional epidemic modeling is **deductive** — hypothesize mechanisms, fit parameters, test
- SINDy flips it to **inductive** — feed raw case data, algorithm returns equations
- Horrocks & Bauch (2020) tested it on **measles / chickenpox / rubella** in the pre-vaccine era
- Two laws automatically rediscovered: mass-action `β·S·I` + seasonal forcing `β(t)·S·I`
- **The question we re-asked**: can a 2022 algorithmic upgrade do better?

---

## Slide 3 — Starting Point

[FIGURE 2]

- Existing Python reimplementation of Horrocks & Bauch (2020), *Sci. Rep.* 10:7061
- Pipeline: smooth → incidence-to-prevalence → susceptible reconstruction → SINDy → grid search
- Real historical data from McMaster IIDDA archive (1946–1967)
- Output state: **19 PNG plots, no machine-readable results**

---

## Slide 4 — Four Weaknesses Identified

1. **No machine-readable output** — coefficients lived in memory only; comparison required OCR
2. **Paper's strongest claim missing** — out-of-sample regime-shift (Fig 8-9) was not reproduced
3. **Three documented overfit / UQ issues**:
   - Chickenpox `β(t)·I² = 114.8` (biologically meaningless, paper authors flag)
   - No confidence intervals on any coefficient
   - Rubella required ad-hoc PSD switch
4. **Heuristic single-point AIC** — fragile on ~10⁻⁴ prevalence data

---

## Slide 5 — Our Chosen Upgrade: Ensemble-SINDy

**Fasel, Kutz, Brunton & Brunton (2022)** — *Proc. Roy. Soc. A* 478:20210904

- Bootstrap data **B = 100** times, run SINDy on each resample
- **Inclusion probability** `P(k)` = fraction of bootstraps where term k survives
- **Median ± 25-75 IQR** per coefficient
- Filter at `P ≥ 0.6`, refit OLS → consensus model

| Plain SINDy | Ensemble SINDy |
|---|---|
| One fit | 100 fits |
| Point estimate | Median + IQR |
| Manual overfit detection | Automatic via P_incl |

---

## Slide 6 — Algorithm in 6 Steps

1. Build library `Θ(X) = [1, S, I, S², I², SI, β·1, β·S, β·I, β·S², β·SI, β·I²]`
2. Resample rows of `(Θ, X_next)` with replacement, **B = 100 times**
3. Run `sparsifyDynamics` on each resample → save `Ξ_b`
4. Compute inclusion probability across all `b`
5. Compute median, `q25`, `q75` per term
6. Filter at `P ≥ 0.6`, refit OLS → consensus `Ξ`

> **Two-pass strategy**: baseline grid first to locate `(S₀*, λ*, φ*)`, then bootstrap in 3×3×3 neighborhood. Total runtime ~30 min for all 3 diseases.

---

## Slide 7 — Pipeline Architecture

**Baseline**
`raw → smooth → preprocess → Θ → sparsifyDynamics ONCE → AIC → single Ξ`

**Ensemble (NEW)**
`raw → [same prep] → Θ → sparsifyDynamics × 100 → P(k) + IQR → filter → CV-AIC → consensus Ξ + uncertainty bands`

Both pipelines run on the same data so paper-vs-baseline-vs-ensemble can be compared directly.

---

## Slide 8 — Engineering Footprint

| File | Status | LOC | Purpose |
|---|---|---|---|
| `src/ensemble_sindy.py` | **NEW** | 330 | Bootstrap + inclusion + filter |
| `src/regime_shift.py` | **NEW** | 220 | Paper Fig 8-9 reproduction |
| `src/results_io.py` | **NEW** | 200 | CSV/JSON export + paper table |
| `src/model_selection.py` | edit | +90 | Forward-chained CV-AIC |
| `src/visualization.py` | edit | +220 | 4 new uncertainty plots |
| `src/config.py` | edit | +25 | New config dicts |
| `run_comparison.py` | **NEW** | 250 | Top-level orchestrator |
| `update_report.py` | **NEW** | 600 | Auto-rebuild updated docx |
| **Total** | | **~1,340** | new/edited Python |

---

## Slide 9 — Win 1: Automatic Overfit Suppression (HEADLINE)

[FIGURE 3]

| Paper says | Our ensemble says |
|---|---|
| `β(t)·I² = +114.765` | **P_incl = 0.09 → filtered out** |
| (biologically meaningless, manually flagged) | (automatic, no human input) |

> The paper's authors had to eyeball this term as overfit. **Our algorithm flagged it without human input.** This is the central E-SINDy promise: spurious terms only survive a fraction of bootstraps.

---

## Slide 10 — Win 2: Uncertainty Quantification

[FIGURE 4]

- **Median + 25-75 IQR** for every coefficient
- **Inclusion probability** = how stable the term is across resamples
- **IQR width** = how precise the magnitude is

**Example**: measles `β(t)·SI` → median **+0.55**, IQR **±0.6**, P_incl **0.61**

Before: a single number. After: a number with an honest confidence statement.

---

## Slide 11 — Win 3: Out-of-Sample Regime-Shift Test

[FIGURE 5]

- Reproduces paper Fig 8-9 — completely **missing** from the original repo
- Perturb S-equation `1` and `S` coefficients by factors `{1.0, 0.75, 0.523, 0.25}`
- Factor **0.523** = paper's vaccination-era recharge drop (0.606 → 0.317)
- Forward-simulate 10 years → power spectrum → dominant period
- **Verification: regime shift detected for all 3 diseases**
- Chickenpox baseline period exactly **1.0 yr** ✓

---

## Slide 12 — Win 4: Quantitative Comparison

[FIGURE 6]

- `results.csv` — paper / baseline / ensemble columns **side-by-side**
- Per-disease JSON — full `Ξ` + bootstrap distribution + inclusion probabilities
- Effect-size deltas now **machine-computable**
- Aligned with TRIPOD-AI / RIGHT-AI healthcare-reproducibility guidelines

---

## Slide 13 — Win 5: Honest Reproducibility Audit

Coefficient magnitudes are ~50× smaller than the paper. We tested 3 hypotheses:

- **H1**: CV-AIC vs in-sample AIC → **FALSIFIED** (still 50× off)
- **H2**: `ΔX` vs `X(t+1)` target → **FALSIFIED** (`β·SI` nearly unchanged)
- **H3**: Reduced-amplitude `β(t)` → **PARTIALLY HELPS** (50× → 15×)

> **Conclusion**: paper's reported coefficients are NOT the OLS optimum on the documented data. Likely a hidden preprocessing step in the original MATLAB. Documented as **Section 11.5** of the report.

Structural findings (which terms survive, signs, suppression) remain valid.

---

## Slide 14 — Comparison Matrix

| Capability | Original repo | Our upgrade |
|---|---|---|
| Algorithm | Plain SINDy (1 fit) | E-SINDy (B=100 bootstraps) |
| Discovered model | Point estimate | Median + IQR + P_incl |
| Spurious-term filter | Manual visual inspection | Automatic at P ≥ 0.6 |
| Model selection | In-sample AIC | Forward-chained CV-AIC |
| Out-of-sample test | Absent | Regime-shift script (Fig 8-9) |
| Result export | PNG only | CSV + JSON + PNG |
| Reproducibility | "Similar plots" | Quantitative audit + falsification |
| Citations | 1 paper | 1 paper + 1 documented upgrade |
| Lines of code | ~2,000 | ~3,340 (+1,340 new) |

---

## Slide 15 — References + Closing

**Take-aways**
- Mass-action + seasonal forcing rediscovered automatically
- Bootstrap aggregation gives discovered ODEs the **uncertainty quantification** healthcare ML demands
- An **honest reproducibility audit** is part of the deliverable, not a weakness

**References**
1. Horrocks J. & Bauch C.T. (2020). Algorithmic discovery of dynamic models from infectious disease data. *Sci. Rep.* 10:7061.
2. Fasel U., Kutz J.N., Brunton B.W., Brunton S.L. (2022). Ensemble-SINDy. *Proc. Roy. Soc. A* 478:20210904.
3. Brunton S.L., Proctor J.L., Kutz J.N. (2016). Discovering governing equations from data. *PNAS* 113:3932.
4. Finkenstädt B.F., Grenfell B.T. (2000). Time series modelling of childhood diseases. *J. R. Stat. Soc. C* 49:187.
5. Mangan N.M. et al. (2017). Model selection for dynamical systems via sparse regression. *Proc. Roy. Soc. A* 473:20170009.

---

## Figure Mapping

All files live in `outputs_real_data/`. Drop each into the matching `[FIGURE N]` slot:

| Slot | Slide | File |
|---|---|---|
| **FIGURE 1** | Slide 2 (Big Question) | `summary_all_diseases.png` |
| **FIGURE 2** | Slide 3 (Starting Point) | `measles_raw_data.png` |
| **FIGURE 3** | Slide 9 (HEADLINE) | `chickenpox_inclusion_heatmap.png` |
| **FIGURE 4** | Slide 10 (UQ) | `measles_coefficients_ensemble.png` |
| **FIGURE 5** | Slide 11 (Regime Shift) | `chickenpox_regime_shift.png` |
| **FIGURE 6** | Slide 12 (Comparison) | `measles_method_comparison.png` |

**Backup figures** (use if a slide needs more than one image, or if a panel asks for the rubella case):
- `rubella_method_comparison.png`
- `rubella_regime_shift.png`
- `measles_inclusion_heatmap.png`
- `measles_regime_shift.png`
