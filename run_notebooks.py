"""
Execute notebooks sequentially using nbconvert.

Run from project root:
    python run_notebooks.py                  # run all of 05-13
    python run_notebooks.py --only 06 07     # run a subset by 2-digit prefix
    python run_notebooks.py --fail-fast      # stop at the first failure

Each notebook runs in its own fresh kernel (restart-safety is checked by
construction). Partially executed notebooks are saved on failure so the
error output is inspectable in the .ipynb itself. A machine-readable summary
is written to .scratch/nb_execution_summary.json after every run.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
except ImportError:
    print("nbconvert not available — installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "nbconvert", "-q"])
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

PROJECT_ROOT = Path(__file__).parent
NB_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "05_feature_engineering.ipynb",
    "06_baseline_models.ipynb",
    "07_survival_models.ipynb",
    "08_multitask_models.ipynb",
    "09_risk_score_construction.ipynb",
    "10_model_interpretability.ipynb",
    "11_baseball_specific_insights.ipynb",
    "12_usage_strategy_simulation.ipynb",
    "13_dashboard.ipynb",
]

TIMEOUT = 7200  # 2 hours per notebook

# The project kernel (Homebrew python3.11 with the full data-science stack:
# pandas, sklearn, xgboost, lifelines, scikit-survival, plotly). The default
# "python3" kernel points at a different interpreter — do not use it unless
# pitcher311 is missing.
def pick_kernel() -> str:
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().find_kernel_specs()
        if "pitcher311" in specs:
            return "pitcher311"
        print("WARNING: pitcher311 kernel not found — falling back to python3")
    except Exception as e:
        print(f"WARNING: could not inspect kernelspecs ({e}) — using python3")
    return "python3"


def parse_args():
    parser = argparse.ArgumentParser(description="Execute project notebooks sequentially.")
    parser.add_argument("--only", nargs="*", metavar="NN",
                        help="run only these notebooks, by 2-digit prefix (e.g. --only 06 07)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop at the first notebook that errors instead of continuing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebooks = NOTEBOOKS
    if args.only:
        notebooks = [nb for nb in NOTEBOOKS if nb[:2] in args.only]
        if not notebooks:
            print(f"No notebooks match prefixes: {args.only}")
            return 2

    kernel = pick_kernel()
    print(f"Executing with kernel: {kernel}")

    results = []
    for nb_name in notebooks:
        nb_path = NB_DIR / nb_name
        print(f"\n{'='*60}")
        print(f"Executing: {nb_name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            with open(nb_path) as f:
                nb = nbformat.read(f, as_version=4)

            ep = ExecutePreprocessor(
                timeout=TIMEOUT,
                kernel_name=kernel,
            )
            ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})

            with open(nb_path, "w") as f:
                nbformat.write(nb, f)

            elapsed = time.time() - t0
            print(f"SUCCESS: {nb_name} completed in {elapsed:.0f}s")
            results.append({"notebook": nb_name, "status": "success", "elapsed": elapsed})

        except CellExecutionError as e:
            elapsed = time.time() - t0
            print(f"ERROR in {nb_name} after {elapsed:.0f}s:")
            print(str(e)[:2000])
            results.append({"notebook": nb_name, "status": "error", "elapsed": elapsed,
                            "error": str(e)[:1000]})
            # Write partially executed notebook
            with open(nb_path, "w") as f:
                nbformat.write(nb, f)
            if args.fail_fast:
                print("Partial execution saved. Stopping (--fail-fast).")
                break
            print("Partial execution saved. Continuing to next notebook...")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FATAL in {nb_name}: {e}")
            results.append({"notebook": nb_name, "status": "fatal", "elapsed": elapsed,
                            "error": str(e)[:1000]})
            if args.fail_fast:
                break

    print(f"\n\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_sym = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_sym} {r['notebook']:45s} {r['status']:8s} {r.get('elapsed', 0):.0f}s")

    # Machine-readable summary for the verifier / autonomous loop
    summary_path = PROJECT_ROOT / ".scratch" / "nb_execution_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary_path.write_text(json.dumps(
        {"run_at": datetime.now(timezone.utc).isoformat(), "results": results},
        indent=2,
    ))
    print(f"\nSummary written to {summary_path}")

    return 0 if all(r["status"] == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
