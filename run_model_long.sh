#!/bin/bash
# Autonomous notebook-completion loop — long-running / credit-unlimited variant.
# Same as run_project.sh but never exits early; waits for notebook runs to finish
# and keeps calling Claude until all notebooks pass or MAX_ITERS is exhausted.
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

PROMPT='Read claude_instructions.md and follow the Session Protocol.

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

for i in $(seq 1 "$MAX_ITERS"); do
    echo ""
    echo "############################################################"
    echo "# Iteration $i of $MAX_ITERS — $(date)"
    echo "############################################################"

    # If a notebook is running, wait for it to finish before calling Claude.
    if ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; then
        echo "Notebook execution is running. Waiting for it to finish..."
        while ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep >/dev/null; do
            sleep 30
        done
        echo "Notebook finished at $(date). Continuing loop."
    fi

    if "$PYTHON" scripts/verify_outputs.py; then
        echo ""
        echo "✓ All notebooks complete and verified after $((i-1)) iteration(s)."
        afplay /System/Library/Sounds/Funk.aiff
        exit 0
    fi

    echo ""
    echo "Verifier reports work remaining — starting Claude session $i..."

    claude -p "$PROMPT" \
        --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(ps *),Bash(grep *),Bash(tail *),Bash(cat *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
        --permission-mode acceptEdits \
        --max-turns 80 \
        --verbose \
        2>&1 | tee "logs/run_$(date +%Y%m%d)_iter${i}.log"

    echo "Claude session $i finished."

    sleep 30
done

echo ""
if "$PYTHON" scripts/verify_outputs.py; then
    echo "✓ All notebooks complete and verified."
    afplay /System/Library/Sounds/Funk.aiff
    exit 0
fi

echo "✗ Did not converge after $MAX_ITERS iterations. See logs/ and .scratch/verification.json."
afplay /System/Library/Sounds/Funk.aiff
exit 1
