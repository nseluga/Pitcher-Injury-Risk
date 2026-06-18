#!/bin/bash
# Autonomous loop — long-running / credit-unlimited variant.
# Same phase logic as run_project.sh but NEVER exits early:
# waits for notebook runs to finish before calling Claude again.
#
# Phases:
#   Phase 1 (fix):       verify_outputs.py fails → fix the first broken notebook.
#   Phase 2 (critique):  verify passes, critiques pending → baseball-research critique & improve.
#   Phase 3A (improve):  all critiques done, improvement not converged → iterative model improvement.
#   Phase 3B (dashboard): improvement converged, dashboard not done → build unified Streamlit dashboard.
#
# Usage:
#   ./run_model_long.sh [max_iterations]   default 50

set -u

MAX_ITERS="${1:-50}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs .scratch

PYTHON="python3"
[ -x "$PROJECT_ROOT/.venv/bin/python" ] && PYTHON="$PROJECT_ROOT/.venv/bin/python"

export BASH_DEFAULT_TIMEOUT_MS=600000
export BASH_MAX_TIMEOUT_MS=14400000

# ---------------------------------------------------------------------------
# PHASE 1 PROMPT
# ---------------------------------------------------------------------------
FIX_PROMPT='Read claude_instructions.md and follow the Phase 1 Session Protocol.

Your job this session:
1. Run scripts/verify_outputs.py, identify the FIRST failing notebook.
2. Debug it, make code/notebook edits, then launch:
   python run_notebooks.py --only NN --fail-fast
3. After launching, STOP IMMEDIATELY. Do not poll.
4. Only verify and commit if the notebook already finished during this session.
5. Update .scratch/progress.json and docs/notebook_debug_log.md.

Before doing anything, check:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

If a real notebook process is running, do not start another one.'

# ---------------------------------------------------------------------------
# PHASE 2 PROMPT
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT='Read claude_instructions.md and follow the Phase 2 Critique Protocol.

All notebooks are passing verify_outputs.py. Critique the next uncritiqued
modeling notebook, implement research-grounded improvements, rerun, verify, commit.

1. cat .scratch/critique_progress.json
   cat docs/model_critique_log.md | tail -80

2. Find the first "pending" notebook. If critique_progress.json does not exist,
   create it and start with NB05.

3. Read the notebook cell by cell. List every key modeling decision.

4. WebSearch 3-5 relevant baseball injury papers.

5. Write a critique section in docs/model_critique_log.md.

6. Implement the highest-value improvements grounded in research.

7. Run TEST_MODE first, then full:
   python run_notebooks.py --only NN --fail-fast

8. python scripts/verify_outputs.py --only NN

9. Set notebook to "done" in .scratch/critique_progress.json.

10. Commit:
    git add notebooks/NN_*.ipynb src/ docs/model_critique_log.md .scratch/critique_progress.json
    git commit -m "NBxx critique: <one-line summary>"

IMPORTANT: Check for running notebook before launching one:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
After launching, STOP IMMEDIATELY.'

# ---------------------------------------------------------------------------
# PHASE 3A PROMPT — model improvement
# ---------------------------------------------------------------------------
IMPROVE_PROMPT='Read claude_instructions.md and follow the Phase 3A Model Improvement Protocol.

The models currently achieve PR-AUC ~0.127–0.137 on the 30-day injury prediction
task (temporal CV, folds 1-4). Your job is to run ONE improvement round this
session: research an idea, implement it, test it, measure the PR-AUC delta, log,
and commit.

1. Orient:
   cat .scratch/improvement_progress.json 2>/dev/null || echo "NOT STARTED"
   cat docs/model_improvement_log.md 2>/dev/null | tail -120
   cat reports/tables/baseline_model_metrics.csv

2. Pick the next untried improvement idea from the Phase 3A section of
   claude_instructions.md. If improvement_progress.json does not exist, create
   it with baseline metrics and status "in_progress".

3. Run 1-2 targeted WebSearches to ground the idea in research evidence.

4. Implement the change (usually NB05, NB06, NB07, or src/).

5. Test:
   python run_notebooks.py --only NN --fail-fast  (TEST_MODE first)
   python run_notebooks.py --only NN --fail-fast  (full run)
   python scripts/verify_outputs.py --only NN

6. Measure PR-AUC delta (temporal CV mean, folds 1-4, 30-day horizon).

7. If the change hurt metrics, revert it:
   git checkout notebooks/NN_*.ipynb

8. Log results in docs/model_improvement_log.md following the format in
   claude_instructions.md.

9. Update .scratch/improvement_progress.json with the round result.
   Increment consecutive_non_improvements if delta < 0.005.

10. If consecutive_non_improvements >= 3 OR rounds_completed >= 10:
    set _meta.status = "converged".

11. Commit:
    git add notebooks/ src/ docs/model_improvement_log.md .scratch/improvement_progress.json reports/
    git commit -m "Phase 3A round N: [description] (PR-AUC +X.XXX)"

IMPORTANT: Check for running notebook before launching one:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
After launching, STOP IMMEDIATELY.'

# ---------------------------------------------------------------------------
# PHASE 3B PROMPT — unified analysis dashboard
# ---------------------------------------------------------------------------
DASHBOARD_PROMPT='Read claude_instructions.md and follow the Phase 3B Dashboard Protocol.

Model improvement is converged. Build (or continue building) a unified Streamlit
analysis dashboard at dashboard/app.py combining the four NB13 prototype components.

1. Orient:
   cat .scratch/dashboard_progress.json 2>/dev/null || echo "NOT STARTED"

2. Check if Streamlit is installed:
   /opt/homebrew/opt/python@3.11/bin/python3.11 -c "import streamlit; print(streamlit.__version__)" 2>/dev/null
   If not: /opt/homebrew/opt/python@3.11/bin/pip3.11 install streamlit

3. Implement dashboard/app.py following the Phase 3B section of
   claude_instructions.md. Port logic from NB13 cells. Use @st.cache_data
   for all data loads. All four panels must work with real data:
   - Season leaderboard with archetype filter
   - Pitcher profile (name search, IR+ trend, component breakdown)
   - Multi-pitcher trend comparison
   - Archetype analysis panel

4. Confirm the app has no import errors or missing data path issues.

5. Update .scratch/dashboard_progress.json:
   {"status": "done", "launch_command": "/opt/homebrew/opt/python@3.11/bin/streamlit run dashboard/app.py", "panels_complete": ["leaderboard", "pitcher_profile", "trend_comparison", "archetype_panel"]}

6. Commit:
   git add dashboard/ .scratch/dashboard_progress.json
   git commit -m "Phase 3B: unified Streamlit analysis dashboard"'

# ---------------------------------------------------------------------------
# HELPER — wait for any running notebook to finish
# ---------------------------------------------------------------------------
wait_for_notebook() {
    if ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; then
        echo "Notebook execution is running. Waiting for it to finish..."
        while ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; do
            sleep 30
        done
        echo "Notebook finished at $(date). Continuing loop."
    fi
}

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

for i in $(seq 1 "$MAX_ITERS"); do
    echo ""
    echo "############################################################"
    echo "# Iteration $i of $MAX_ITERS — $(date)"
    echo "############################################################"

    wait_for_notebook

    # ----------------------------------------------------------------
    # Determine current phase
    # ----------------------------------------------------------------

    if ! "$PYTHON" scripts/verify_outputs.py 2>&1 | tee "logs/verify_$(date +%Y%m%d)_iter${i}.log"; then
        # ------- Phase 1: notebooks failing -------
        echo ""
        echo "Verifier reports failures — starting Phase 1 fix session $i..."
        claude -p "$FIX_PROMPT" \
            --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
            --permission-mode acceptEdits \
            --max-turns 80 \
            --verbose \
            2>&1 | tee "logs/run_$(date +%Y%m%d)_iter${i}.log"
        echo "Claude fix session $i finished."

    else
        # All notebooks passing — determine sub-phase.

        # Phase 2: pending critiques?
        CRITIQUE_JSON=".scratch/critique_progress.json"
        PENDING="unknown"
        if [ -f "$CRITIQUE_JSON" ]; then
            PENDING=$(python3 -c "
import json, sys
data = json.load(open('$CRITIQUE_JSON'))
pending = [k for k,v in data.items() if v.get('status') == 'pending']
print(len(pending))
" 2>/dev/null || echo "unknown")
        fi

        if [ "$PENDING" != "0" ] && [ "$PENDING" != "" ]; then
            # ------- Phase 2: critique -------
            echo ""
            echo "Phase 2 critique pending ($PENDING remaining) — starting critique session $i..."
            claude -p "$CRITIQUE_PROMPT" \
                --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,WebSearch,WebFetch,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                --permission-mode acceptEdits \
                --max-turns 80 \
                --verbose \
                2>&1 | tee "logs/critique_$(date +%Y%m%d)_iter${i}.log"
            echo "Claude critique session $i finished."

        else
            # Phase 2 done. Check Phase 3A convergence.
            IMPROVE_JSON=".scratch/improvement_progress.json"
            IMPROVE_STATUS="not_started"
            if [ -f "$IMPROVE_JSON" ]; then
                IMPROVE_STATUS=$(python3 -c "
import json, sys
data = json.load(open('$IMPROVE_JSON'))
print(data.get('_meta', {}).get('status', 'in_progress'))
" 2>/dev/null || echo "in_progress")
            fi

            if [ "$IMPROVE_STATUS" != "converged" ]; then
                # ------- Phase 3A: model improvement -------
                echo ""
                echo "Phase 3A model improvement (status: $IMPROVE_STATUS) — starting improvement session $i..."
                claude -p "$IMPROVE_PROMPT" \
                    --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,WebSearch,WebFetch,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                    --permission-mode acceptEdits \
                    --max-turns 80 \
                    --verbose \
                    2>&1 | tee "logs/improve_$(date +%Y%m%d)_iter${i}.log"
                echo "Claude improvement session $i finished."

                # Wait for any notebook that was launched in this session.
                wait_for_notebook

            else
                # Phase 3A converged. Check Phase 3B (dashboard).
                DASH_JSON=".scratch/dashboard_progress.json"
                DASH_STATUS="not_started"
                if [ -f "$DASH_JSON" ]; then
                    DASH_STATUS=$(python3 -c "
import json, sys
data = json.load(open('$DASH_JSON'))
print(data.get('status', 'not_started'))
" 2>/dev/null || echo "not_started")
                fi

                if [ "$DASH_STATUS" != "done" ]; then
                    # ------- Phase 3B: dashboard -------
                    echo ""
                    echo "Phase 3B dashboard (status: $DASH_STATUS) — starting dashboard session $i..."
                    claude -p "$DASHBOARD_PROMPT" \
                        --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pip *),Bash(pip3 *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                        --permission-mode acceptEdits \
                        --max-turns 80 \
                        --verbose \
                        2>&1 | tee "logs/dashboard_$(date +%Y%m%d)_iter${i}.log"
                    echo "Claude dashboard session $i finished."

                else
                    # ------- All done -------
                    echo ""
                    echo "✓ All phases complete: models improved and dashboard built."
                    afplay /System/Library/Sounds/Funk.aiff
                    exit 0
                fi
            fi
        fi
    fi

    sleep 30
done

echo ""
echo "✗ Did not converge after $MAX_ITERS iterations. See logs/ and .scratch/."
afplay /System/Library/Sounds/Funk.aiff
exit 1
