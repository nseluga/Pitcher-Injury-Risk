# Project Status — Pitcher Injury Risk+

Last updated: 2026-06-17

---

## Pipeline Phase

**Phase 1 — Implementation: COMPLETE**  
All 13 notebooks implemented, executed, and verified end-to-end on the full 2015–2024 dataset.

**Phase 2 — Baseball Research Critique & Improvement: COMPLETE**  
All modeling notebooks 05–12 critiqued against published baseball injury literature. Improvements applied, re-run, and verified. Full record in `docs/model_critique_log.md`. All critique progress entries show `"status": "done"` in `.scratch/critique_progress.json`.

**Phase 3 — Interpretation of Findings: IN PROGRESS**  
The pipeline is stable and research-grounded. The current task is to synthesize what the models found into clear, defensible conclusions. Output target: `docs/findings_summary.md`. See `claude_instructions.md` for the full Phase 3 protocol.

---

## Pipeline Overview

| # | Notebook | Phase 1 | Phase 2 Critique |
|---|----------|---------|-----------------|
| 01 | `01_data_collection.ipynb` | ✅ | — |
| 02 | `02_injury_database_construction.ipynb` | ✅ | — |
| 03 | `03_data_cleaning.ipynb` | ✅ | — |
| 04 | `04_eda.ipynb` | ✅ | — |
| 05 | `05_feature_engineering.ipynb` | ✅ | ✅ Done |
| 06 | `06_baseline_models.ipynb` | ✅ | ✅ Done |
| 07 | `07_survival_models.ipynb` | ✅ | ✅ Done |
| 08 | `08_multitask_models.ipynb` | ✅ | ✅ Done |
| 09 | `09_risk_score_construction.ipynb` | ✅ | ✅ Done |
| 10 | `10_model_interpretability.ipynb` | ✅ | ✅ Done |
| 11 | `11_baseball_specific_insights.ipynb` | ✅ | ✅ Done |
| 12 | `12_usage_strategy_simulation.ipynb` | ✅ | ✅ Done |
| 13 | `13_dashboard.ipynb` | ✅ | — |

**Legend:**
- ✅ Complete (full) — implemented, executed on full 2015–2024 data, verified
- 🔲 Pending — Phase 2 critique not yet run
- ✅ (Phase 2) — critique complete, improvements applied and verified

---

## Phase 1 Completion Summary

All artifacts from notebooks 01–13 are on disk. Key statistics from the full dataset:

- **Statcast data:** ~4.5M pitches, 2015–2024 (10 seasons)
- **Pitcher-appearances in feature matrix:** ~206K rows
- **Injury labels (30d):** ~13–14% positive rate
- **Baseline XGBoost (tuned) PR-AUC:** see `reports/tables/tuned_baseline_model_metrics.csv`
- **Survival Cox C-index:** see `reports/tables/survival_model_metrics.csv`
- **Injury Risk+ score:** mean ≈ 100 (by construction), `data/processed/injury_risk_plus_scores.parquet`

---

## Critical Architecture Notes

- **Python environment:** Notebooks use `pitcher311` kernel → `/opt/homebrew/opt/python@3.11/bin/python3.11`. The `.venv/bin/python` (3.13) drives `run_project.sh` and verifier only.
- **Memory safety:** All modeling notebooks use subsampling / batching. Peak RAM ~4–8 GB for Cox/RSF fitting.
- **Top feature by importance:** `prior_il_total` (0.26 XGBoost importance) dominates — injury history is the strongest predictor, as expected from baseball research.

---

## Artifacts on Disk

### `data/raw/`
- `statcast/` — per-season parquets, 2015–2024
- `injuries/injury_database.parquet` — full IL history 2015–2024
- `player_metadata/pitchers.parquet` — all pitchers with birth dates

### `data/processed/`
- `feature_matrix.parquet` — ~206K rows × 88 cols (76 features + labels + metadata)
- `feature_matrix_train/val/test.parquet` — temporal splits
- `features/` — 5 per-group parquets (workload, velocity, pitch_mix, movement, injury_history)
- `injury_risk_plus_scores.parquet` — Injury Risk+ scores for all pitcher-appearances

### `models/`
- `baseline_logistic.joblib`, `baseline_logistic_tuned.joblib`
- `baseline_random_forest.joblib`, `baseline_random_forest_tuned.joblib`
- `baseline_xgboost.joblib`, `baseline_xgboost_tuned.joblib`
- `survival_cox.pkl`, `survival_rsf.pkl`
- `multitask_chained.joblib`, `multitask_chained_tuned.joblib`

### `reports/`
- `figures/` — EDA plots, SHAP, PDP, Risk+ distribution, simulation charts, dashboard HTML
- `tables/` — baseline/survival/multitask metrics, tuning results, leaderboard, simulation results

---

## Phase 3 Immediate Next Steps

1. Read the key output artifacts: `reports/tables/shap_global_importance.csv`, `reports/tables/baseball_specific_insights_summary.csv`, `reports/tables/injury_risk_plus_leaderboard.csv`, and the per-year top-25 tables.
2. Work through the six interpretation targets in order (top predictors → score distribution → historical leaderboards → archetype differences → simulation insights → limitations).
3. Write findings to `docs/findings_summary.md`, one section at a time.
4. Commit after each section with `git commit -m "Phase 3: [section] interpretation complete"`.

See `claude_instructions.md` for the full Phase 3 protocol and quality bar.

---

## Source Modules

| Module | Status |
|--------|--------|
| `src/data/statcast_loader.py` | ✅ Complete |
| `src/data/injury_loader.py` | ✅ Complete |
| `src/features/workload_features.py` | ✅ Complete |
| `src/features/velocity_features.py` | ✅ Complete |
| `src/features/pitch_mix_features.py` | ✅ Complete |
| `src/features/movement_features.py` | ✅ Complete |
| `src/features/injury_history_features.py` | ✅ Complete |
| `src/models/baseline_models.py` | ✅ Complete |
| `src/models/survival_models.py` | ✅ Complete |
| `src/models/multitask_models.py` | ✅ Complete |
| `src/models/evaluation.py` | ✅ Complete |
| `src/scoring/injury_risk_plus.py` | ✅ Complete |
| `src/scoring/score_calibration.py` | ✅ Complete |
| `src/simulation/workload_simulator.py` | ✅ Complete |
| `src/simulation/pitch_mix_simulator.py` | ✅ Complete |
| `src/simulation/usage_strategy_simulator.py` | ✅ Complete |
