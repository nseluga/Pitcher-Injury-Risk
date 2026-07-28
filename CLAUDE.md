# CLAUDE.md

Guidance for working on the Pitcher Injury Risk+ project.

## What this project is

A baseball pitcher health modeling platform that predicts injury probability,
estimates severity, models time to injury, and produces a normalized composite
score called Injury Risk+. See `README.md` for the full research framing and
`docs/` for design documents.

## Repository layout

- `notebooks/` — the numbered analysis pipeline (01 to 13), run in order.
- `src/` — importable modules: data loaders, feature engineering, models,
  scoring, simulation, visualization.
- `scripts/` — helper scripts for notebook generation and output verification.
- `dashboard/` — Streamlit app that combines the analysis views into one tool.
- `docs/` — data dictionary, roadmap, design notes, and research questions.
- `models/`, `reports/`, `data/` — generated artifacts, not committed.

## Environment

- Conda environment name: `pitcher-injury-risk` (Python 3.11).
- Create it with `conda env create -f environment.yml`, or install with
  `pip install -r requirements.txt`.

## Running the pipeline

```bash
python run_notebooks.py            # run notebooks 05 to 13 in fresh kernels
python run_notebooks.py --only 06  # run a subset by 2-digit prefix
python scripts/verify_outputs.py   # check that expected outputs exist
```

Data is ingested programmatically. No raw or processed datasets are committed,
so notebooks 01 to 04 must run first to populate `data/`.

## Methodology constraints

These rules protect the validity of the models and must always hold.

- No leakage. Post-injury data must never appear in features.
- No causal claims. All counterfactuals are model based, not causal.
- Any resampling such as SMOTE is applied inside cross-validation folds only,
  never on the full dataset before splitting.
- Do not silently drop pitchers or seasons from the training set.
- If a change breaks `scripts/verify_outputs.py`, revert it.

## Code and notebook standards

Notebooks follow a readability-first style. Each logical group of cells gets a
Markdown heading named after what it produces, followed by one or two plain
sentences explaining what it does and why. Print results as plain labeled
lines rather than decorative banners, and keep comments to short English notes
above the code they describe.
