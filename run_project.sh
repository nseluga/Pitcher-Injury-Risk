```bash
#!/bin/bash
# Autonomous notebook-completion loop.
#
# Goal:
# - Use Claude for diagnosis + code edits.
# - Use your laptop for long notebook execution.
# - Prevent Claude from wasting quota by waiting/polling.
#
# Usage:
#   ./run_project.sh [max_iterations]   default 8

set -u

MAX_ITERS="${1:-8}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs .scratch

PYTHON="python3"
[ -x "$PROJECT_ROOT/.venv/bin/python" ] && PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Give Claude enough time for real commands, but the prompt tells it not to
# babysit long notebook runs.
export BASH_DEFAULT_TIMEOUT_MS=600000      # 10 min default
export BASH_MAX_TIMEOUT_MS=14400000       # 4 hr ceiling

PROMPT='Read claude_instructions.md and follow the Session Protocol.

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

    if "$PYTHON" scripts/verify_outputs.py; then
        echo ""
        echo "✓ All notebooks complete and verified after $((i-1)) iteration(s)."
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

    # If Claude launched a notebook, stop the outer loop so the laptop can run alone.
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
```
