#!/bin/bash
# Autonomous notebook loop — two phases:
#
#   Phase 1 (fix):     verify_outputs.py fails → fix the first broken notebook.
#   Phase 2 (critique): verify_outputs.py passes → baseball-research critique &
#                        improve each modeling notebook in order.
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

export BASH_DEFAULT_TIMEOUT_MS=600000      # 10 min default
export BASH_MAX_TIMEOUT_MS=14400000       # 4 hr ceiling

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
   references. Specific queries to try:
   - For NB05: "ACWR acute chronic workload ratio baseball pitchers injury"
   - For NB06: "machine learning pitcher injury prediction class imbalance"
   - For NB07: "cox proportional hazards baseball pitcher survival analysis"
   - For NB08: "injury severity prediction baseball days missed distribution"
   - For NB09: "injury risk score normalization era adjustment baseball"
   - For NB10: "SHAP pitcher injury features importance validation"
   - For NB11: "slider pitch UCL injury risk pitcher mechanics"
   - For NB12: "pitcher workload reduction injury risk counterfactual"
   Adapt queries based on what you actually find in the notebook.

5. Write a critique section in docs/model_critique_log.md following the format
   in the Phase 2 Critique Protocol in claude_instructions.md.

6. Implement the highest-value improvements. Quality bar: grounded in research,
   measurably changes a metric or closes a literature gap, does not break anything.

7. Run TEST_MODE first to verify the notebook runs without errors:
   python run_notebooks.py --only NN --fail-fast
   (with TEST_MODE=True in the notebook or a small sample)

8. Run full if TEST_MODE passes:
   python run_notebooks.py --only NN --fail-fast

9. Verify: python scripts/verify_outputs.py --only NN

10. Update .scratch/critique_progress.json — set the notebook to "done".

11. Commit:
    git add notebooks/NN_*.ipynb src/ docs/model_critique_log.md .scratch/critique_progress.json
    git commit -m "NBxx critique: <one-line summary of main improvement>"

12. If you have capacity and the session is not long, move to the next pending notebook.

IMPORTANT: do NOT run a notebook without first checking for a running process:
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

IMPORTANT: After launching a notebook, STOP IMMEDIATELY. Do not poll.'

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

    # Determine phase.
    if "$PYTHON" scripts/verify_outputs.py 2>&1 | tee "logs/verify_$(date +%Y%m%d)_iter${i}.log"; then
        # ----------------------------------------------------------------
        # Phase 2: all notebooks pass — run critique/improve loop
        # ----------------------------------------------------------------

        # Check if all notebooks have been critiqued.
        CRITIQUE_JSON=".scratch/critique_progress.json"
        if [ -f "$CRITIQUE_JSON" ]; then
            PENDING=$(python3 -c "
import json, sys
data = json.load(open('$CRITIQUE_JSON'))
pending = [k for k,v in data.items() if v.get('status') == 'pending']
print(len(pending))
" 2>/dev/null || echo "unknown")
            if [ "$PENDING" = "0" ]; then
                echo ""
                echo "✓ Phase 2 complete: all modeling notebooks have been critiqued and improved."
                exit 0
            fi
        fi

        echo ""
        echo "Phase 1 complete. Entering Phase 2 critique session $i..."

        claude -p "$CRITIQUE_PROMPT" \
            --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,WebSearch,WebFetch,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
            --permission-mode acceptEdits \
            --max-turns 80 \
            --verbose \
            2>&1 | tee "logs/critique_$(date +%Y%m%d)_iter${i}.log"

        echo "Claude critique session $i finished."

    else
        # ----------------------------------------------------------------
        # Phase 1: notebooks failing — fix loop
        # ----------------------------------------------------------------
        echo ""
        echo "Verifier reports failures — starting Phase 1 fix session $i..."

        claude -p "$FIX_PROMPT" \
            --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
            --permission-mode acceptEdits \
            --max-turns 80 \
            --verbose \
            2>&1 | tee "logs/run_$(date +%Y%m%d)_iter${i}.log"

        echo "Claude fix session $i finished."
    fi

    # If Claude launched a notebook, exit so the laptop can run it alone.
    if ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; then
        echo "Notebook execution launched. Exiting so it can run without burning Claude quota."
        ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
        exit 0
    fi

    sleep 30
done

echo ""
if "$PYTHON" scripts/verify_outputs.py; then
    echo "✓ All notebooks complete and verified."
    exit 0
fi

echo "✗ Did not converge after $MAX_ITERS iterations. See logs/ and .scratch/verification.json."
exit 1
