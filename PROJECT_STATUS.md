# Project Status — Pitcher Injury Risk+

Last updated: 2026-06-07

---

## Pipeline Overview

| # | Notebook | Status | Data Mode | Outputs on Disk |
|---|----------|--------|-----------|----------------|
| 01 | `01_data_collection.ipynb` | ✅ Complete (test) | TEST_MODE | Statcast + metadata parquet |
| 02 | `02_injury_database_construction.ipynb` | ✅ Complete (test) | TEST_MODE | Injury database parquet |
| 03 | `03_data_cleaning.ipynb` | ✅ Complete (test) | TEST_MODE | Cleaned parquets |
| 04 | `04_eda.ipynb` | ✅ Complete (test) | TEST_MODE | 19 figures + feature ranking |
| 05 | `05_feature_engineering.ipynb` | ✅ Complete (test) | TEST_MODE | Feature matrix (718 rows, 76 features) |
| 06 | `06_baseline_models.ipynb` | 🔲 Not started | — | None |
| 07 | `07_survival_models.ipynb` | 🔲 Not started | — | None |
| 08 | `08_multitask_models.ipynb` | 🔲 Not started | — | None |
| 09 | `09_risk_score_construction.ipynb` | 🔲 Not started | — | None |
| 10 | `10_model_interpretability.ipynb` | 🔲 Not started | — | None |
| 11 | `11_baseball_specific_insights.ipynb` | 📋 Placeholder | — | None |
| 12 | `12_usage_strategy_simulation.ipynb` | 🔲 Not started | — | None |
| 13 | `13_dashboard.ipynb` | 🔲 Not started | — | None |

**Legend:**
- ✅ Complete (test) — implemented and executed, but on 1-week test data only
- 🔲 Not started — import stub + TODO only; no implementation
- 📋 Placeholder — structure and code scaffolding written; not executed

---

## Critical Blocker

> **All notebooks 01–05 were run in `TEST_MODE = True`.**
> This pulls only one week of Statcast data (2023-06-01 → 2023-06-07) instead of the full
> 2015–2024 historical dataset. All model-ready artifacts on disk reflect this limited scope.
>
> **Before running notebook 06 or beyond, set `TEST_MODE = False` in notebook 01 and
> re-run the full pipeline.** The full pull takes approximately 90 minutes.

---

## Notebook Detail

### 01 — Data Collection ✅ Complete (test)

**Last run:** 2026-06-07T04:19:01 UTC  
**Data scope:** 2023-06-01 → 2023-06-07 (1 week, test mode)

| Artifact | Status | Details |
|----------|--------|---------|
| `data/raw/statcast/range_20230601_20230607.parquet` | ✅ Present | 25,714 pitches, 80 columns |
| `data/raw/player_metadata/pitchers.parquet` | ✅ Present | 400 pitchers, birth dates from MLB Stats API |
| `data/raw/provenance.json` | ✅ Present | Pipeline run metadata |

**Notes:**
- pybaseball cache enabled; re-runs skip already-downloaded seasons
- Birth dates fetched from MLB Stats API (all 400 pitchers resolved)
- Full pull (2015–2024) will produce ~10 season files under `data/raw/statcast/`

---

### 02 — Injury Database Construction ✅ Complete (test)

**Last run:** 2026-06-07T04:19:15 UTC  
**Data scope:** 2023 IL transactions only (test mode)

| Artifact | Status | Details |
|----------|--------|---------|
| `data/raw/injuries/injury_database.parquet` | ✅ Present | 244 IL stints, 180 pitchers |

**Key metrics:**
- IL placements parsed: 244 (all pitchers, 2023)
- Median days lost: 24 days
- Mean days lost: 38 days
- `other` injury type: 17 stints (7.0%) — irreducible, no description in API
- Season-ending stints (no activation): 34 (13.9%)
- Stints under 10 days flagged: 11 (retroactive end-of-season moves)

**Parser unit tests:** All 9 passing

---

### 03 — Data Cleaning ✅ Complete (test)

**Last run:** 2026-06-07T04:19:45 UTC

| Artifact | Status | Details |
|----------|--------|---------|
| `data/processed/statcast_clean.parquet` | ✅ Present | 25,613 rows (101 implausible removed) |
| `data/processed/injuries_clean.parquet` | ✅ Present | 244 stints + `retroactive_short` flag |
| `data/processed/player_metadata_clean.parquet` | ✅ Present | 400 pitchers, 0 null birth dates |

**Cleaning steps applied:**
- Dropped 2 always-null columns (`spin_dir`, `sv_id`)
- Deduplicated on `(game_pk, at_bat_number, pitch_number)` — 0 duplicates found
- Pitch type remapping: FT→SI, FA→FF, FO→FS, CS→CU; removed PO/IN/AB
- Removed 101 rows with implausible spin rates (<1000 rpm)
- Birth dates supplemented from MLB Stats API (all 400 resolved)
- Cross-source join validation: 100% Statcast-metadata coverage

---

### 04 — Exploratory Data Analysis ✅ Complete (test)

**Last run:** 2026-06-07T04:20:22 UTC

| Artifact | Status | Details |
|----------|--------|---------|
| `data/processed/pitcher_eda_features.parquet` | ✅ Present | 400 pitchers, pitcher-season grain |
| `data/processed/feature_ranking_mi.parquet` | ✅ Present | MI scores for 23 features |
| `data/processed/feature_ranking_mi.csv` | ✅ Present | Human-readable copy |
| `data/processed/pitcher_clusters.parquet` | ✅ Present | 5 pitcher archetypes (KMeans) |
| `reports/figures/fig_01_missing_values.png` through `fig_19_archetype_injury_rates.png` | ✅ Present | 19 figures |

**Key EDA findings (test data — treat as provisional):**
- Pitchers analyzed: 400
- Future injury rate: 37.3% (inflated — reflects only 6 days of pre-injury observation)
- Top features by mutual information: `velo_delta`, `avg_velo`, `avg_extension`, `max_velo`, `std_velo`
- Pitcher archetypes found: 5 (KMeans on velocity + pitch mix + movement)
- ACWR: degenerate in test mode (requires ≥28 days of history)
- Injury countdown: placeholder only (insufficient longitudinal data in test window)

**Figures generated:** 19 (missing value analysis → archetype injury rates)

**⚠ Test mode caveats:**
- Injury rate of 37% is an artifact of the 1-week observation window
- Rolling features (ACWR, velocity decline) are mostly null — need full season data
- Countdown analysis (pre-injury trajectory) requires multi-week history

---

### 05 — Feature Engineering ✅ Complete (test)

**Last run:** 2026-06-07T03:32:24 UTC

| Artifact | Status | Details |
|----------|--------|---------|
| `data/processed/feature_matrix.parquet` | ✅ Present | 718 rows × 88 columns |
| `data/processed/feature_matrix_train.parquet` | ✅ Present | 718 rows (single season, all in train) |
| `data/processed/feature_manifest.json` | ✅ Present | 76 features + 6 label columns listed |

**Feature groups built:**
| Group | Features | Notes |
|-------|----------|-------|
| Workload | 7 | `pitch_count`, `pitches_7/28/90d`, `acwr_7_28`, `days_rest`, `pitches_season_to_date` |
| Velocity | 15 | Mean, max, std, rolling 7/30/90d avgs, change metrics, season delta |
| Pitch mix | 17 | Usage rates + 30d rolling averages + 30d deltas for FB, SL, CU, CH, breaking |
| Movement | 21 | pfx, spin, extension, release point position + drift metrics |
| Injury history | 16 | `prior_il_total`, per-type counts, `days_since_last_injury`, `prior_il_days_lost` |

**Labels built:**
- `injured_next_30d`: 98 positive (13.6%)
- `injured_next_60d`: 147 positive (20.5%)
- `injured_next_90d`: 208 positive (29.0%)

**⚠ Test mode caveats:**
- 87% null rate on `days_since_last_injury` (only 1 week of pitcher history)
- 86% null rate on `release_drift_30d` (rolling features need history)
- 56% null rate on `fb_velo_30d_avg`, `velo_change_7_30d`, `days_rest`
- Train/val/test split not possible with a single season — all 718 rows in train set
- No validation or test split saved to disk

---

### 06 — Baseline Models 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned models:**
- Logistic regression (L2 regularized)
- Random forest with cross-validation
- XGBoost / LightGBM
- Naive baseline (historical rate by archetype)

**Expected outputs:**
- `models/baseline_logistic.joblib`
- `models/baseline_random_forest.joblib`
- `models/baseline_xgboost.joblib`
- `reports/tables/baseline_model_results.csv`

**Blocking:** Requires full 2015–2024 feature matrix for meaningful train/val/test split.

---

### 07 — Survival Models 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned models:** Cox PH, Weibull AFT, Random Survival Forest

**Blocking:** Requires full longitudinal data; temporal structure is meaningless with 1-week window.

---

### 08 — Multi-Task Models 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned:** Chained multi-task (probability → type → severity) and shared-representation model.

**Blocking:** Requires baseline models from notebook 06 as a benchmark.

---

### 09 — Risk Score Construction 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned:** Probability calibration, blend weight optimization, era normalization, score leaderboards.

**Blocking:** Requires trained models from notebooks 06–08.

---

### 10 — Model Interpretability 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned:** SHAP global/local, partial dependence plots, feature interaction analysis.

**Blocking:** Requires trained models from notebooks 06–08.

---

### 11 — Baseball-Specific Insights 📋 Placeholder

**Status:** Full structure written with working placeholder code; not executed.

**Sections:**
1. Load data and merge Risk+ scores
2. Risk+ distribution
3. Risk+ vs pitch mix (slider, breaking ball, fastball, changeup)
4. Risk+ vs velocity (level, change, spike)
5. Risk+ vs workload (pitches, ACWR)
6. Risk+ vs rest and recovery
7. Risk interaction heatmaps (5 planned: velocity×slider, velocity×workload, slider×rest, ACWR×history, age×workload)
8. Pitcher archetype risk (rule-based fallback if archetypes not available)
9. Role-based risk
10. Performance vs risk frontier
11. Pre-injury risk trajectory (Day −90 to Day 0)
12. Novel insight candidate list
13. Conclusions

**Expected outputs:**
- 11 figures under `reports/figures/`
- `reports/tables/baseball_specific_insights_summary.csv`

**Blocking:** Requires Injury Risk+ scores from notebook 09.

---

### 12 — Usage Strategy Simulation 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned:** Pitch count simulator, rest schedule optimizer, pitch mix simulator, role transition analysis.

**Blocking:** Requires trained models from notebook 09 and insights from notebook 11.

---

### 13 — Dashboard 🔲 Not started

**Status:** Import stub only. No implementation.

**Planned:** Streamlit or Dash prototype with pitcher lookup, leaderboard, and trend explorer.

**Blocking:** Requires Injury Risk+ scores and simulation results from notebooks 09–12.

---

## Artifacts on Disk

### `data/raw/`

| File | Size info | Notes |
|------|-----------|-------|
| `statcast/range_20230601_20230607.parquet` | 25,714 rows | Test window only |
| `injuries/injury_database.parquet` | 244 stints | 2023 season only |
| `player_metadata/pitchers.parquet` | 400 rows | Birth dates resolved |
| `provenance.json` | — | Pipeline run log |

### `data/processed/`

| File | Rows | Notes |
|------|------|-------|
| `statcast_clean.parquet` | 25,613 | After dedup + implausible removal |
| `injuries_clean.parquet` | 244 | After reclassification + short-stint flagging |
| `player_metadata_clean.parquet` | 400 | Birth dates filled |
| `feature_matrix.parquet` | 718 × 88 | 76 features + 6 labels + metadata |
| `feature_matrix_train.parquet` | 718 | All rows in train (single season) |
| `pitcher_eda_features.parquet` | 400 | Pitcher-season grain from NB04 |
| `pitcher_clusters.parquet` | 400 | 5 KMeans archetypes |
| `feature_ranking_mi.parquet/.csv` | 23 | MI + Pearson correlation ranking |
| `feature_manifest.json` | — | 76-feature column manifest |

### `models/`
*Empty — no trained models yet.*

### `reports/figures/`
19 EDA figures (`fig_01_missing_values.png` through `fig_19_archetype_injury_rates.png`).  
All from notebook 04, all based on 1-week test data.

### `reports/tables/`
*Empty — no summary tables yet.*

---

## Immediate Next Steps

1. **Re-run notebook 01 with `TEST_MODE = False`** — this initiates the full 2015–2024 Statcast pull (~90 min). All subsequent notebooks depend on this.
2. **Re-run notebooks 02–05** after the full data pull to regenerate clean artifacts with the complete dataset.
3. **Implement notebook 06** (baseline models) once full feature matrix is available.
4. Continue sequentially through notebooks 07 → 08 → 09 → 10 → 11 → 12 → 13.

---

## Source Modules

| Module | Status | Notes |
|--------|--------|-------|
| `src/data/statcast_loader.py` | ✅ Complete | Used by NB01 |
| `src/data/injury_loader.py` | ✅ Complete | Used by NB02–03 |
| `src/features/workload_features.py` | ✅ Complete | Used by NB05 |
| `src/features/velocity_features.py` | ✅ Complete | Used by NB05 |
| `src/features/pitch_mix_features.py` | ✅ Complete | Used by NB05 |
| `src/features/movement_features.py` | ✅ Complete | Used by NB05 |
| `src/features/injury_history_features.py` | ✅ Complete | Used by NB05 |
| `src/models/baseline_models.py` | 🔲 Stub | Imported in NB06; not implemented |
| `src/models/survival_models.py` | 🔲 Stub | Imported in NB07; not implemented |
| `src/models/multitask_models.py` | 🔲 Stub | Imported in NB08; not implemented |
| `src/models/evaluation.py` | 🔲 Stub | Imported in NB06; not implemented |
| `src/scoring/injury_risk_plus.py` | 🔲 Stub | Imported in NB09; not implemented |
| `src/scoring/score_calibration.py` | 🔲 Stub | Imported in NB09; not implemented |
| `src/simulation/workload_simulator.py` | 🔲 Stub | Imported in NB12; not implemented |
| `src/simulation/pitch_mix_simulator.py` | 🔲 Stub | Imported in NB12; not implemented |
| `src/simulation/usage_strategy_simulator.py` | 🔲 Stub | Imported in NB12; not implemented |
