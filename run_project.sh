#!/bin/bash
# Autonomous notebook loop — three phases:
#
#   Phase 1 (fix):       verify_outputs.py fails → fix the first broken notebook.
#   Phase 2 (critique):  verify_outputs.py passes, critiques pending → research critique & improve.
#   Phase 3A (improve):  all critiques done, improvement not converged → iterative model improvement.
#   Phase 3B (dashboard): improvement converged, dashboard not done → build unified Streamlit dashboard.
#
# Usage:
#   ./run_project.sh [max_iterations]   default 12

set -u

MAX_ITERS="${1:-12}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs .scratch

PYTHON="python3"
[ -x "$PROJECT_ROOT/.venv/bin/python" ] && PYTHON="$PROJECT_ROOT/.venv/bin/python"

export BASH_DEFAULT_TIMEOUT_MS=600000
export BASH_MAX_TIMEOUT_MS=14400000

# ---------------------------------------------------------------------------
# PHASE 1 PROMPT — fix the first failing notebook
# ---------------------------------------------------------------------------
FIX_PROMPT='Read claude_instructions.md and follow the Phase 1 Session Protocol.

IMPORTANT: do NOT wait around for long-running notebook execution.
Your job this session is one of these:

1. If a notebook is actively running, record that in .scratch/progress.json and STOP.
2. If no notebook is running, run scripts/verify_outputs.py, identify the FIRST failing notebook, debug it, make code/notebook edits, then launch:
   python run_notebooks.py --only NN --fail-fast
   only if the fix is ready.
3. After launching a long notebook run, STOP IMMEDIATELY. Do not poll repeatedly. Do not say "I will wait." Do not burn turns watching it.
4. Only verify and commit if the notebook execution has already finished during this session.
5. Work only on the FIRST failing notebook. Do not move to later notebooks.
6. Update .scratch/progress.json and docs/notebook_debug_log.md with what you changed or launched.

Before doing anything, check:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

If a real notebook process is running, do not start another one.'

# ---------------------------------------------------------------------------
# PHASE 2 PROMPT — baseball-research critique and improve
# ---------------------------------------------------------------------------
CRITIQUE_PROMPT='Read claude_instructions.md and follow the Phase 2 Critique Protocol.

All notebooks 05-13 are currently passing verify_outputs.py. Your job is to
critique the modeling decisions in the next uncritiqued notebook, improve them
based on baseball injury research, rerun, verify, and commit.

Step-by-step:

1. Orient:
   cat .scratch/critique_progress.json
   cat docs/model_critique_log.md | tail -80

2. Identify the first notebook whose status is "pending" in critique_progress.json.
   If critique_progress.json does not exist yet, create it with all notebooks set
   to "pending" and start with NB05.

3. Read the target notebook cell by cell. List every key modeling decision.

4. Use WebSearch to research 3-5 relevant baseball injury papers or industry
   references. Adapt queries based on what you actually find in the notebook.

5. Write a critique section in docs/model_critique_log.md.

6. Implement the highest-value improvements grounded in research.

7. Run TEST_MODE first, then full:
   python run_notebooks.py --only NN --fail-fast

8. Verify: python scripts/verify_outputs.py --only NN

9. Update .scratch/critique_progress.json — set the notebook to "done".

10. Commit:
    git add notebooks/NN_*.ipynb src/ docs/model_critique_log.md .scratch/critique_progress.json
    git commit -m "NBxx critique: <one-line summary of main improvement>"

IMPORTANT: do NOT run a notebook without first checking for a running process:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

IMPORTANT: After launching a notebook, STOP IMMEDIATELY. Do not poll.'

# ---------------------------------------------------------------------------
# PHASE 3A PROMPT — survival model improvement
# ---------------------------------------------------------------------------
IMPROVE_PROMPT='Read claude_instructions.md and follow the Phase 3A Survival Model Improvement Protocol.

The binary classifiers have plateaued. The goal now is to find useful signal in
the survival models (NB07) — models that answer *when* a pitcher is likely to get
injured, not just whether. Current C-index is ~0.514 (near-random). Your job is
to run ONE improvement round this session: brainstorm, research, implement, test,
measure C-index delta, log, and commit.

Step-by-step:

1. Orient — read what has already been tried:
   cat .scratch/improvement_progress.json 2>/dev/null || echo "NOT STARTED"
   cat docs/model_improvement_log.md 2>/dev/null | tail -120

2. Read notebooks/07_survival_models.ipynb cell by cell to understand the
   current survival model setup.

3. Write a brainstorm list of 8-10 new approaches to
   .scratch/survival_improvement_ideas.md (see Step 0 format in
   claude_instructions.md). This step is REQUIRED before picking an idea.

4. Pick the highest-expected-value idea not yet attempted from your brainstorm
   list. If improvement_progress.json does not exist, create it with baseline
   C-index 0.514 and status "in_progress".

5. Run 1-2 targeted WebSearches to ground the idea in evidence.

6. Implement the change in notebooks/07_survival_models.ipynb and/or
   src/models/survival_models.py. Keep it targeted — one idea per round.

7. Test:
   python run_notebooks.py --only 07 --fail-fast  (TEST_MODE first)
   python run_notebooks.py --only 07 --fail-fast  (full run)
   python scripts/verify_outputs.py --only 07

8. Measure C-index and IBS delta. Note any interpretable survival curves
   or hazard ratios — these are wins even if C-index gain is small.

9. If the change hurt metrics, revert it:
   git checkout notebooks/07_survival_models.ipynb src/models/survival_models.py

10. Log results in docs/model_improvement_log.md following the format in
    claude_instructions.md.

11. Update .scratch/improvement_progress.json with the round result and
    increment consecutive_non_improvements if C-index delta < 0.005.

12. If consecutive_non_improvements >= 3 OR rounds_completed >= 10:
    set _meta.status = "converged" in improvement_progress.json.

13. Commit:
    git add notebooks/07_survival_models.ipynb src/models/survival_models.py docs/model_improvement_log.md .scratch/improvement_progress.json .scratch/survival_improvement_ideas.md
    git commit -m "Phase 3A round N: [description] (C-index +X.XXX)"

IMPORTANT: do NOT run a notebook without first checking for a running process:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

IMPORTANT: After launching a notebook, STOP IMMEDIATELY. Do not poll.'

# ---------------------------------------------------------------------------
# PHASE 3B PROMPT — unified analysis dashboard
# ---------------------------------------------------------------------------
DASHBOARD_PROMPT='Read claude_instructions.md and follow the Phase 3B Dashboard Protocol.

Model improvement is converged. Your job is to build (or continue building) a
unified Streamlit analysis dashboard at dashboard/app.py that combines all four
NB13 prototype components into a genuinely usable tool.

Step-by-step:

1. Orient:
   cat .scratch/dashboard_progress.json 2>/dev/null || echo "NOT STARTED"
   python3 -c "
import json
nb = json.load(open(\"notebooks/13_dashboard.ipynb\"))
for i, cell in enumerate(nb[\"cells\"]):
    src = \"\".join(cell[\"source\"])[:100].replace(\"\\n\",\" \")
    print(f\"[{i}] {cell[\"cell_type\"]}: {src}\")
"

2. Check if Streamlit is installed for the pitcher311 environment:
   /opt/homebrew/opt/python@3.11/bin/python3.11 -c "import streamlit; print(streamlit.__version__)" 2>/dev/null
   If not: /opt/homebrew/opt/python@3.11/bin/pip3.11 install streamlit

3. Implement dashboard/app.py following the Phase 3B section of
   claude_instructions.md. Port logic from NB13 cells. Use @st.cache_data
   for all data loads. All four panels must work with real data.

4. Confirm the app launches without errors (check for import errors, missing
   data paths, etc.) by examining the code logic. Do not spin up a server
   process during this session.

5. Update .scratch/dashboard_progress.json:
   {"status": "done", "launch_command": "/opt/homebrew/opt/python@3.11/bin/streamlit run dashboard/app.py", "panels_complete": [...]}

6. Commit:
   git add dashboard/ .scratch/dashboard_progress.json
   git commit -m "Phase 3B: unified Streamlit analysis dashboard"'

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

for i in $(seq 1 "$MAX_ITERS"); do
    echo ""
    echo "############################################################"
    echo "# Iteration $i of $MAX_ITERS — $(date)"
    echo "############################################################"

    # If a notebook is already running, do not start Claude.
    if ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; then
        echo "Notebook execution is already running. Not starting Claude."
        ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
        exit 0
    fi

    # ----------------------------------------------------------------
    # Phase 1 vs Phase 2+ gating
    # ----------------------------------------------------------------
    if ! "$PYTHON" scripts/verify_outputs.py 2>&1 | tee "logs/verify_$(date +%Y%m%d)_iter${i}.log"; then
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
        # All notebooks passing — determine which sub-phase to run.

        # Phase 2: pending critiques?
        CRITIQUE_JSON=".scratch/critique_progress.json"
        if [ -f "$CRITIQUE_JSON" ]; then
            PENDING=$(python3 -c "
import json, sys
data = json.load(open('$CRITIQUE_JSON'))
pending = [k for k,v in data.items() if v.get('status') == 'pending']
print(len(pending))
" 2>/dev/null || echo "unknown")
        else
            PENDING="unknown"
        fi

        if [ "$PENDING" != "0" ] && [ "$PENDING" != "" ]; then
            echo ""
            echo "Phase 2 critique pending — starting critique session $i..."
            claude -p "$CRITIQUE_PROMPT" \
                --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,WebSearch,WebFetch,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                --permission-mode acceptEdits \
                --max-turns 80 \
                --verbose \
                2>&1 | tee "logs/critique_$(date +%Y%m%d)_iter${i}.log"
            echo "Claude critique session $i finished."

        else
            # Phase 2 done. Check Phase 3A (improvement) convergence.
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
                echo ""
                echo "Phase 3A model improvement — starting improvement session $i..."
                claude -p "$IMPROVE_PROMPT" \
                    --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,WebSearch,WebFetch,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                    --permission-mode acceptEdits \
                    --max-turns 80 \
                    --verbose \
                    2>&1 | tee "logs/improve_$(date +%Y%m%d)_iter${i}.log"
                echo "Claude improvement session $i finished."

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
                    echo ""
                    echo "Phase 3B dashboard — starting dashboard session $i..."
                    claude -p "$DASHBOARD_PROMPT" \
                        --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pip *),Bash(pip3 *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
                        --permission-mode acceptEdits \
                        --max-turns 80 \
                        --verbose \
                        2>&1 | tee "logs/dashboard_$(date +%Y%m%d)_iter${i}.log"
                    echo "Claude dashboard session $i finished."

                else
                    echo ""
                    echo "✓ All phases complete: models improved and dashboard built."
                    afplay /System/Library/Sounds/Funk.aiff
                    exit 0
                fi
            fi
        fi
    fi

    # If Claude launched a notebook, exit so it can run without burning Claude quota.
    if ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; then
        echo "Notebook execution launched. Exiting so it can run without burning Claude quota."
        ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
        exit 0
    fi

    sleep 30
done

echo ""
echo "✗ Did not converge after $MAX_ITERS iterations. See logs/ and .scratch/."
exit 1
