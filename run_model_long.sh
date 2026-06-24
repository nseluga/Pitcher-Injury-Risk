#!/bin/bash
# Sequential bug-fix pass: run NB02–NB13 in order, fix any bugs found, then stop.
#
# Skips NB01 (already works, very long runtime).
# For each failing notebook: spawns Claude to fix bugs, then retries until it passes.
# Exits 0 when all notebooks pass.
#
# Usage: ./run_model_long.sh

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs .scratch

PYTHON="python3"
[ -x "$PROJECT_ROOT/.venv/bin/python" ] && PYTHON="$PROJECT_ROOT/.venv/bin/python"

LOG_TS=$(date +%Y%m%d_%H%M%S)
ATTEMPT=0
NOTEBOOKS=(02 03 04 05 06 07 08 09 10 11 12 13)

# ---------------------------------------------------------------------------
# run_notebook <NN>: execute via run_notebooks.py. Returns 0/1.
# ---------------------------------------------------------------------------
run_notebook() {
    local nb_num="$1"
    local err_file=".scratch/nb_${nb_num}_error.txt"

    echo ""
    echo "--- Running NB${nb_num} ---"
    local rc
    "$PYTHON" run_notebooks.py --only "$nb_num" --fail-fast 2>&1 | tee "$err_file"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        echo "NB${nb_num}: SUCCESS"
        rm -f "$err_file"
        return 0
    else
        echo "NB${nb_num}: FAILED (exit $rc) — error saved to $err_file"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# fix_notebook <NN>: spawn Claude to identify and fix the bug.
# ---------------------------------------------------------------------------
fix_notebook() {
    local nb_num="$1"
    local nb_file
    nb_file=$(ls "notebooks/${nb_num}_"*.ipynb 2>/dev/null | head -1)
    local err_file=".scratch/nb_${nb_num}_error.txt"
    local fix_log="logs/fix_nb${nb_num}_${LOG_TS}.log"

    local FIX_PROMPT
    FIX_PROMPT="You are fixing a bug in a Jupyter notebook for a baseball pitcher injury risk prediction project.

FAILING NOTEBOOK: $nb_file
ERROR FILE: $err_file

Steps:
1. Read the error output: cat $err_file
2. Read the notebook (Read tool) and any relevant src/ files to understand the failure.
3. Identify the root cause — which cell fails and exactly why.
4. Make the minimal fix in the notebook cell(s) and/or src/ files.
   - Do NOT refactor, restructure, or improve code beyond the bug.
   - Fix imports rather than adding pip install cells.
   - If a column name or file path is wrong, update the code to match the actual data.
5. Run the notebook to verify:
   $PYTHON run_notebooks.py --only $nb_num --fail-fast
6. If it still fails, read the new error and iterate (up to 3 more cycles).
7. Once it passes, commit:
   git add $nb_file src/
   git commit -m 'Fix NB${nb_num}: <one-line description of bug fixed>'

PIPELINE CONTEXT:
NB01=data_collection (already run, outputs in data/)
NB02=injury_DB_construction → NB03=cleaning → NB04=EDA →
NB05=feature_engineering → NB06=baseline_models → NB07=survival_models →
NB08=multitask_models → NB09=risk_score → NB10=interpretability →
NB11=baseball_insights → NB12=usage_simulation → NB13=dashboard

Start by orienting yourself:
ls data/ 2>/dev/null | head -30
ls src/ 2>/dev/null"

    echo ""
    echo "Spawning Claude to fix NB${nb_num} (log: $fix_log)..."

    claude -p "$FIX_PROMPT" \
        --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(cat *),Bash(head *),Bash(tail *),Bash(python *),Bash(python3 *),Bash($PYTHON *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
        --permission-mode acceptEdits \
        --max-turns 30 \
        --verbose \
        2>&1 | tee "$fix_log"

    echo "Claude fix session for NB${nb_num} done."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Notebook Bug-Fix Pass  NB02 → NB13  $(date)"
echo "========================================================"

for nb_num in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  NB${nb_num}  $(date)"
    echo "========================================================"

    ATTEMPT=0
    while ! run_notebook "$nb_num"; do
        ATTEMPT=$((ATTEMPT + 1))
        echo ""
        echo "NB${nb_num}: fix attempt $ATTEMPT"
        fix_notebook "$nb_num"
        echo "NB${nb_num}: re-running after fix..."
    done
done

echo ""
echo "========================================================"
echo "  All notebooks NB02–NB13 passed successfully.  $(date)"
echo "========================================================"
afplay /System/Library/Sounds/Funk.aiff 2>/dev/null || true
exit 0
