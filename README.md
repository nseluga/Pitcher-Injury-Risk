# Pitcher Injury Risk+

A baseball pitcher health modeling platform designed to predict injury probability, estimate severity, model time-to-injury, and produce a normalized composite risk score — **Injury Risk+** — that supports both predictive analysis and novel usage strategy research.

---

## Motivation

Pitcher injuries are among the most costly and disruptive events in professional baseball. Despite advances in biomechanics and sports science, ML-driven pitcher health modeling remains largely siloed, focusing narrowly on binary injury prediction rather than taking a holistic view of pitcher health over time.

This project approaches pitcher health as a multi-dimensional research problem:

- **When** will a pitcher likely get injured?
- **How severe** will the injury be?
- **How long** will recovery take?
- **What factors** are most predictive of risk?
- **How can usage strategies be changed** to proactively reduce risk?

The answers to these questions have direct applications for team roster construction, workload management, and long-term pitcher development.

---

## Research Objectives

1. Build a reliable **injury probability model** trained on Statcast pitch-level data, workload metrics, velocity trends, and pitch mix features.
2. Construct **injury severity** and **expected days lost** models to complement binary injury prediction.
3. Develop a **time-to-injury survival model** using accelerated failure time (AFT) and Cox proportional hazards approaches.
4. Produce a single, interpretable **Injury Risk+** composite score for each pitcher at each time step.
5. Achieve **model interpretability** via SHAP values and permutation importance to surface which factors drive risk.
6. Build **simulation infrastructure** that allows researchers to evaluate alternative pitcher usage strategies and discover novel risk-reduction approaches.

---

## Injury Risk+ Concept

**Injury Risk+** is a normalized composite score modeled after ERA+ and OPS+ in offensive analytics:

| Score | Interpretation |
|-------|---------------|
| 100   | League-average injury risk for this pitcher archetype |
| > 100 | Higher injury risk than average |
| < 100 | Lower injury risk than average |

The score is computed by:

1. Predicting raw injury probability from the trained ensemble.
2. Adjusting for **pitcher archetype** (starter vs. reliever vs. hybrid), **age**, and **injury history**.
3. Normalizing so that the population mean equals 100 in each season.
4. Calibrating so that a 10-point increase in IR+ corresponds to a meaningful, interpretable increase in actual injury probability.

This design allows Injury Risk+ to be used for:

- Season-level risk profiling
- Rolling risk monitoring throughout a season
- Cross-season comparisons
- Scenario analysis (e.g., "What happens to this pitcher's IR+ if we reduce slider usage by 10%?")

---

## Data Sources

| Source | Contents | Access Method |
|--------|----------|---------------|
| [PyBaseball](https://github.com/jldbc/pybaseball) | Statcast pitch-level data, FanGraphs, Baseball Reference | Python library |
| [MLB Stats API](https://statsapi.mlb.com) | Game logs, roster transactions, player metadata | REST API |
| [Baseball Savant](https://baseballsavant.mlb.com) | Statcast search, leaderboards, expected stats | Scraping + PyBaseball |
| MLB Transactions | IL placements, activations, 60-day IL | REST API + scraping |
| Injury Databases | Historical injury records | To be integrated |

All data is ingested programmatically via code — no static datasets are committed to the repository.

---

## Repository Structure

```
Pitcher-Injury-Risk/
│
├── data/
│   ├── raw/              # Unmodified API/scrape outputs
│   │   ├── statcast/
│   │   ├── injuries/
│   │   ├── transactions/
│   │   └── player_metadata/
│   ├── processed/        # Cleaned, merged, feature-engineered datasets
│   └── external/         # Third-party reference data
│
├── notebooks/            # Numbered analysis notebooks (sequential pipeline)
├── src/
│   ├── data/             # Data loaders and master dataset builder
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model training, evaluation, survival models
│   ├── scoring/          # Injury Risk+ score construction and calibration
│   ├── simulation/       # Workload, pitch mix, and strategy simulators
│   ├── visualization/    # Plotting utilities
│   └── utils/            # Shared helpers
│
├── models/               # Serialized trained models
├── reports/              # Figures and tables for publication/presentation
├── tests/                # Unit and integration tests
└── docs/                 # Architecture docs, data dictionary, roadmap
```

---

## Modeling Roadmap

### Phase 1 — Foundation
- [x] Project scaffolding and architecture
- [ ] Data ingestion pipelines (Statcast, transactions, MLB Stats API)
- [ ] Injury database construction from transactions
- [ ] Data cleaning and validation

### Phase 2 — Feature Engineering
- [ ] Workload features (pitch count, innings, rest, rolling windows)
- [ ] Velocity features (trends, spikes, decline curves)
- [ ] Movement features (spin rate, break, release point drift)
- [ ] Pitch mix features (usage rates, mix entropy, pitch type shifts)
- [ ] Injury history features (prior IL stints, severity, return timing)

### Phase 3 — Baseline Modeling
- [ ] Logistic regression injury probability baseline
- [ ] Random forest baseline
- [ ] Gradient boosting (XGBoost/LightGBM) baseline
- [ ] Calibration and evaluation framework

### Phase 4 — Advanced Modeling
- [ ] Cox proportional hazards (time-to-injury)
- [ ] Accelerated failure time models
- [ ] Multi-task learning (joint injury/severity/days lost prediction)
- [ ] Recurrent models (LSTM/GRU for longitudinal sequences)

### Phase 5 — Scoring and Interpretability
- [ ] Injury Risk+ composite score construction
- [ ] Score normalization and calibration
- [ ] SHAP-based feature attribution
- [ ] Partial dependence and interaction effects

### Phase 6 — Baseball-Specific Insights
- [ ] Risk vs. pitch mix, velocity, workload, and rest analysis
- [ ] Two-dimensional risk interaction heatmaps (danger zones)
- [ ] Pitcher archetype and role-based risk profiles
- [ ] Pre-injury risk trajectory analysis
- [ ] Novel insight candidate table

### Phase 7 — Simulation and Optimization
- [ ] Workload simulator
- [ ] Pitch mix simulator
- [ ] Full usage strategy optimizer

---

## Simulation and Optimization Roadmap

The simulation layer allows researchers to ask counterfactual questions:

- **What if a pitcher threw 10 fewer pitches per start?**
- **What if slider usage dropped from 35% to 25%?**
- **What if a starter moved to a bulk reliever role?**
- **What is the optimal rest schedule for a high-velocity pitcher over age 30?**

The simulator takes a trained model and a proposed usage policy, generates synthetic workload trajectories, and returns predicted IR+ scores and injury probability distributions over a projected season.

---

## Future Research Questions

See [docs/future_research_questions.md](docs/future_research_questions.md) for the full list of planned research investigations.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/nseluga/Pitcher-Injury-Risk.git
cd Pitcher-Injury-Risk

# Create conda environment
conda env create -f environment.yml
conda activate pitcher-injury-risk

# Or install via pip
pip install -r requirements.txt
```

---

## License

MIT

---

## Author

Nate Seluga
