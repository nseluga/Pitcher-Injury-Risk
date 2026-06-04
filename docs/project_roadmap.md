# Project Roadmap: Pitcher Injury Risk+

## Overview

This document tracks the development phases of the Pitcher Injury Risk+ modeling platform. Each phase builds on the prior one and is designed so that earlier phases produce independently useful artifacts before the full system is complete.

---

## Phase 1: Data Infrastructure

**Goal:** Build a reproducible data pipeline that ingests all necessary raw data from public APIs and assembles a clean, analysis-ready master dataset.

**Milestones:**
- [ ] Statcast pitch-level data collection (2015–present) via pybaseball
- [ ] IL transaction database construction (placement + activation pairs, days lost)
- [ ] Broader transaction history (role changes, rehab assignments)
- [ ] Player metadata (age, physical build, throws, debut date)
- [ ] Game-level aggregation pipeline
- [ ] Master dataset assembly with injury labels
- [ ] Data validation and quality checks

**Outputs:**
- `data/processed/master_dataset.parquet`
- `data/raw/` populated with versioned raw files
- `notebooks/01_data_collection.ipynb` and `02_injury_database_construction.ipynb` fully executed

---

## Phase 2: Exploratory Data Analysis & Feature Engineering

**Goal:** Understand the data, identify patterns, and engineer a rich feature set that captures the known and hypothesized drivers of pitcher injury.

**Milestones:**
- [ ] EDA notebook: distributions, missing data, correlations
- [ ] Workload feature pipeline (pitch counts, rolling windows, rest days)
- [ ] Velocity feature pipeline (mean, delta, spikes)
- [ ] Movement and release point features
- [ ] Pitch mix features (usage rates, entropy, deltas)
- [ ] Injury history features (recurrence, recency, severity)
- [ ] Risk factor aggregation (player attributes, schedule density)
- [ ] Feature correlation analysis and importance screening

**Outputs:**
- `notebooks/04_eda.ipynb` and `05_feature_engineering.ipynb` fully executed
- `data/processed/feature_matrix.parquet`
- `reports/figures/` populated with EDA charts

---

## Phase 3: Baseline Models

**Goal:** Establish performance benchmarks using interpretable, well-understood models before moving to more complex approaches.

**Milestones:**
- [ ] Binary injury classification (30 / 60 / 90-day windows)
- [ ] Days-lost regression
- [ ] Injury type multiclass classification
- [ ] Naive baselines (historical rates by archetype)
- [ ] Temporal cross-validation framework
- [ ] Model evaluation suite (AUC-ROC, PR-AUC, Brier, calibration)

**Outputs:**
- `models/baseline_*.joblib`
- `notebooks/06_baseline_models.ipynb`
- `reports/tables/baseline_model_results.csv`

---

## Phase 4: Survival Models

**Goal:** Frame injury prediction as a time-to-event problem, producing survival functions and hazard rates for individual pitchers.

**Milestones:**
- [ ] Cox Proportional Hazards model
- [ ] Accelerated Failure Time (Weibull, log-normal)
- [ ] Random Survival Forest
- [ ] C-index and integrated Brier score evaluation
- [ ] Survival curve visualization by pitcher archetype

**Outputs:**
- `models/survival_*.pkl`
- `notebooks/07_survival_models.ipynb`

---

## Phase 5: Multi-Task Models

**Goal:** Jointly predict injury probability, injury type, severity, and time-to-event to leverage shared information across related outcomes.

**Milestones:**
- [ ] Chained multi-task model (probability → type → severity)
- [ ] Shared-representation multi-output model
- [ ] Multi-task evaluation framework
- [ ] Comparison against single-task baselines

**Outputs:**
- `models/multitask_*.joblib`
- `notebooks/08_multitask_models.ipynb`

---

## Phase 6: Injury Risk+ Score

**Goal:** Distill model outputs into a single, interpretable, normalized score that can be communicated to non-technical stakeholders.

**Milestones:**
- [ ] Blend weight optimization
- [ ] Probability calibration
- [ ] Era and archetype normalization
- [ ] Score validation (does it predict future outcomes?)
- [ ] Historical score reconstruction (2015–present)
- [ ] Score leaderboards and visual summaries

**Outputs:**
- `models/risk_plus_calibration.pkl`
- `notebooks/09_risk_score_construction.ipynb`
- `reports/figures/risk_plus_leaderboard.png`

---

## Phase 7: Model Interpretability

**Goal:** Understand what drives the model's predictions to build trust, validate domain knowledge, and surface novel insights.

**Milestones:**
- [ ] SHAP global and local explanations
- [ ] Partial dependence plots for key features
- [ ] Feature interaction analysis
- [ ] Case studies: high-risk vs. low-risk pitcher profiles

**Outputs:**
- `notebooks/10_model_interpretability.ipynb`
- `reports/figures/shap_*.png`

---

## Phase 8: Usage Strategy Simulation

**Goal:** Answer counterfactual and optimization questions using the trained models to simulate alternative usage strategies.

**Milestones:**
- [ ] Pitch count reduction simulator
- [ ] Rest schedule optimizer
- [ ] Pitch mix simulator (slider reduction, mix diversification)
- [ ] Role transition simulator (starter → hybrid, reliever)
- [ ] Staff-level workload redistribution
- [ ] Archetype comparison (traditional starter vs. hybrid)

**Outputs:**
- `notebooks/11_usage_strategy_simulation.ipynb`
- `reports/tables/simulation_results.csv`

---

## Phase 9: Dashboard

**Goal:** Build an interactive dashboard for exploring pitcher risk scores, trends, and simulation results.

**Milestones:**
- [ ] Pitcher lookup and risk profile view
- [ ] Season leaderboard with filters
- [ ] Trend charts (Injury Risk+ over time)
- [ ] Simulation interface (adjust workload, see risk change)
- [ ] Deployment plan (Streamlit / Dash / static site)

**Outputs:**
- `notebooks/12_dashboard.ipynb`
- Deployed dashboard (TBD)
