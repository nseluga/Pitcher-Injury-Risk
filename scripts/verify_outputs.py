"""
Verify that notebooks 05-13 are complete: every required artifact exists and
every notebook has executed top-to-bottom with no error outputs.

Run from project root: python scripts/verify_outputs.py
Limit scope:           python scripts/verify_outputs.py --only 06 07

Exit code 0 means the pipeline is done. Anything else means there is work
left — the autonomous loop in run_project.sh uses this as its stop condition.

This manifest is the contract for "done". If a notebook's design legitimately
changes what it outputs, update the manifest here AND record the reason in
docs/notebook_debug_log.md in the same session.
"""

import argparse
import glob as globlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# REQUIRED ARTIFACTS PER NOTEBOOK
# ---------------------------------------------------------------------------
# Paths are relative to project root. Entries containing "*" are glob
# patterns and pass when at least one match exists. An empty list means the
# notebook only needs to execute cleanly (no file contract).

REQUIRED = {
    "05_feature_engineering.ipynb": [
        "data/processed/features/workload_features.parquet",
        "data/processed/features/velocity_features.parquet",
        "data/processed/features/pitch_mix_features.parquet",
        "data/processed/features/movement_features.parquet",
        "data/processed/features/injury_history_features.parquet",
        "data/processed/feature_matrix.parquet",
    ],
    "06_baseline_models.ipynb": [
        "models/baseline_logistic.joblib",
        "models/baseline_random_forest.joblib",
        "models/baseline_xgboost.joblib",
        "models/baseline_logistic_tuned.joblib",
        "models/baseline_random_forest_tuned.joblib",
        "models/baseline_xgboost_tuned.joblib",
        "reports/tables/baseline_model_metrics.csv",
        "reports/tables/hyperparameter_tuning_results.csv",
        "reports/tables/tuned_baseline_model_metrics.csv",
    ],
    "07_survival_models.ipynb": [
        "models/survival_cox.pkl",
        "models/survival_rsf.pkl",
        "reports/tables/survival_model_metrics.csv",
        "reports/tables/survival_hyperparameter_tuning_results.csv",
    ],
    "08_multitask_models.ipynb": [
        "models/multitask_chained.joblib",
        "models/multitask_chained_tuned.joblib",
        "reports/tables/multitask_model_metrics.csv",
        "reports/tables/multitask_hyperparameter_tuning_results.csv",
        "reports/tables/tuned_multitask_model_metrics.csv",
    ],
    "09_risk_score_construction.ipynb": [
        "data/processed/injury_risk_plus_scores.parquet",
        "reports/tables/injury_risk_plus_leaderboard.csv",
        "reports/tables/risk_score_component_summary.csv",
        "reports/tables/risk_score_model_sources.csv",
        "reports/figures/injury_risk_plus_distribution.png",
        "reports/figures/risk_score_components.png",
    ],
    "10_model_interpretability.ipynb": [
        "reports/figures/shap_global_importance.png",
        "reports/figures/shap_beeswarm.png",
        "reports/figures/partial_dependence_*.png",
    ],
    "11_baseball_specific_insights.ipynb": [],
    "12_usage_strategy_simulation.ipynb": [
        "reports/tables/simulation_results.csv",
        "reports/figures/pitch_count_optimization.png",
    ],
    "13_dashboard.ipynb": [],
}


def check_notebook_execution(nb_path: Path) -> list[str]:
    """Return a list of problems with the notebook's execution state."""
    problems = []
    if not nb_path.exists():
        return [f"notebook missing: {nb_path.name}"]

    nb = json.loads(nb_path.read_text())
    code_cells = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]

    if not code_cells:
        return ["notebook has no code cells"]
    if len(code_cells) <= 1:
        problems.append("notebook is still a stub (1 code cell)")

    unexecuted = sum(1 for c in code_cells if not c.get("execution_count"))
    if unexecuted:
        problems.append(f"{unexecuted}/{len(code_cells)} code cells never executed")

    errors = [
        out.get("ename", "error")
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    if errors:
        problems.append(f"error outputs present: {', '.join(errors)}")

    return problems


def check_artifacts(paths: list[str]) -> list[str]:
    """Return the subset of required paths that are missing (or unmatched globs)."""
    missing = []
    for rel in paths:
        if "*" in rel:
            if not globlib.glob(str(PROJECT_ROOT / rel)):
                missing.append(f"{rel} (no glob matches)")
        elif not (PROJECT_ROOT / rel).exists():
            missing.append(rel)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify notebook pipeline completion.")
    parser.add_argument("--only", nargs="*", metavar="NN",
                        help="check only these notebooks, by 2-digit prefix (e.g. --only 06 07)")
    args = parser.parse_args()

    targets = REQUIRED
    if args.only:
        targets = {k: v for k, v in REQUIRED.items() if k[:2] in args.only}
        if not targets:
            print(f"No notebooks match prefixes: {args.only}")
            return 2

    report = {}
    all_pass = True
    print(f"{'='*64}\nPIPELINE VERIFICATION — {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n{'='*64}")

    for nb_name, artifacts in targets.items():
        problems = check_notebook_execution(NB_DIR / nb_name)
        problems += check_artifacts(artifacts)
        status = "pass" if not problems else "fail"
        all_pass &= status == "pass"
        report[nb_name] = {"status": status, "problems": problems}

        symbol = "✓" if status == "pass" else "✗"
        print(f"  {symbol} {nb_name:42s} {status.upper()}")
        for p in problems:
            print(f"      - {p}")

    out_path = PROJECT_ROOT / ".scratch" / "verification.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(
        {"checked_at": datetime.now(timezone.utc).isoformat(),
         "all_pass": all_pass,
         "notebooks": report},
        indent=2,
    ))

    print(f"\n{'ALL CHECKS PASS' if all_pass else 'INCOMPLETE — see problems above'}")
    print(f"Detail written to {out_path.relative_to(PROJECT_ROOT)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
