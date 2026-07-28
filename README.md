# Pitcher Injury Risk+

I built Injury Risk+ to go past the usual binary "will this pitcher get hurt" question. It combines a calibrated injury-probability model, a survival (time-to-injury) model, and expected days lost into one score, modeled after ERA+ and OPS+: normalized to a population mean of 100 within each season and pitcher archetype (starter / reliever / hybrid). 120 means 20% above the risk baseline for that archetype-season; 80 means 20% below.

## Headline result

Discrimination is genuinely hard here: temporal-CV mean AUC-ROC **0.571** (tuned random forest, held-out 2023–24 test: 0.579), survival-ensemble C-index **0.566**. I'm stating that plainly rather than around it: pitcher injury has a real information ceiling, and that's part of what the project found, not something to bury.

## Data & method

- Statcast pitch-level data (via [pybaseball](https://github.com/jldbc/pybaseball)) joined to an injury database I built from MLB Stats API IL transactions: 205,911 pitcher-game rows, 3,249 pitchers, 2015–2024.
- 76 features across workload (rolling 7/28/90-day pitch counts, ACWR), velocity trend, movement/release-point drift, pitch mix, and injury history.
- Temporal train/test split (train 2015–2022, test 2023–2024) with isotonic calibration and walk-forward CV, not random k-fold; injury data leaks across time otherwise.
- Four model tracks: calibrated classifiers (LR/RF/XGBoost), survival models (Cox PH, AFT, Random Survival Forest, gradient-boosted survival ensemble), multi-task injury/severity/days-lost models, and the Injury Risk+ composite scorer.

## Concrete findings

- A chronic velocity decline past 2 mph roughly doubles observed 30-day injury rate, 6% to 12%, the cleanest dose-response result in the project.
- Prior IL history and 90-day cumulative pitch load dominate the signal; the standard ACWR ratio is weaker than the literature suggests it should be.
- Interventional SHAP, checked against naive path-dependent SHAP, reorders feature importance meaningfully, a check most injury-risk projects skip.
- Isotonic calibration tightens the probability estimates substantially (ECE 0.059 → 0.029), even though raw discrimination doesn't move.

## What's built

Data pipeline, all four model tracks, Injury Risk+ scoring and calibration, SHAP interpretability, a counterfactual usage-strategy simulator (I call out the survivorship-bias caveats directly in the code), and an interactive Streamlit dashboard, all committed, not planned.

All data is ingested programmatically via code; no static datasets are committed to the repository.

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
