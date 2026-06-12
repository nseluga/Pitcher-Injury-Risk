"""Execute notebooks 05-09 sequentially using nbconvert."""
import sys, time
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
]

TIMEOUT = 7200  # 2 hours per notebook

results = []
for nb_name in NOTEBOOKS:
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
            kernel_name="python3",
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
        results.append({"notebook": nb_name, "status": "error", "elapsed": elapsed, "error": str(e)[:500]})
        # Write partially executed notebook
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)
        print(f"Partial execution saved. Continuing to next notebook...")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"FATAL in {nb_name}: {e}")
        results.append({"notebook": nb_name, "status": "fatal", "elapsed": elapsed, "error": str(e)})

print(f"\n\n{'='*60}")
print("EXECUTION SUMMARY")
print(f"{'='*60}")
for r in results:
    status_sym = "✓" if r["status"] == "success" else "✗"
    print(f"  {status_sym} {r['notebook']:45s} {r['status']:8s} {r.get('elapsed', 0):.0f}s")

# Write summary to file
summary_path = PROJECT_ROOT / ".scratch" / "nb_execution_summary.txt"
summary_path.parent.mkdir(exist_ok=True)
with open(summary_path, "w") as f:
    for r in results:
        f.write(f"{r['notebook']}: {r['status']} ({r.get('elapsed',0):.0f}s)\n")
        if "error" in r:
            f.write(f"  ERROR: {r['error'][:200]}\n")
print(f"\nSummary written to {summary_path}")
