#!/bin/bash
# Autonomous notebook-completion loop.
#
# Each iteration: run the verifier; if everything passes, stop. Otherwise
# invoke Claude to fix/implement whatever the verifier says is missing, then
# verify again. A fresh session per iteration means context limits or a
# stalled session can never kill the run — the verifier + progress.json +
# debug log carry state between iterations.
#
# Usage: ./run_project.sh [max_iterations]   (default 8)

set -u

MAX_ITERS="${1:-8}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs .scratch

# Use the project venv if present so nbconvert/sklearn resolve correctly
PYTHON="python3"
[ -x "$PROJECT_ROOT/.venv/bin/python" ] && PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Headless Claude caps Bash commands at 2 min by default — far too short for
# full notebook runs. Raise the ceiling so `python run_notebooks.py` can run
# in the foreground for up to 4 hours.
export BASH_DEFAULT_TIMEOUT_MS=600000      # 10 min default
export BASH_MAX_TIMEOUT_MS=14400000       # 4 hr ceiling

PROMPT='Read claude_instructions.md and follow the Session Protocol at the top of it.
In short: run scripts/verify_outputs.py first, read .scratch/progress.json and
docs/notebook_debug_log.md, then work ONLY on the first failing notebook until
it passes verification. Commit after each notebook passes. Update progress.json
and the debug log as you go. Do not declare the task done yourself — the outer
loop re-runs the verifier and decides.'

for i in $(seq 1 "$MAX_ITERS"); do
    echo ""
    echo "############################################################"
    echo "# Iteration $i of $MAX_ITERS — $(date)"
    echo "############################################################"

    if "$PYTHON" scripts/verify_outputs.py; then
        echo ""
        echo "✓ All notebooks complete and verified after $((i-1)) iteration(s)."
        exit 0
    fi

    echo ""
    echo "Verifier reports work remaining — starting Claude session $i..."

    claude -p "$PROMPT" \
        --allowedTools "Read,Write,Edit,NotebookEdit,Glob,Grep,TodoWrite,Bash(ls *),Bash(find *),Bash(mkdir *),Bash(python *),Bash(python3 *),Bash(.venv/bin/python *),Bash(/opt/homebrew/opt/python@3.11/bin/python3.11 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter *),Bash(git status*),Bash(git diff*),Bash(git log*),Bash(git add *),Bash(git commit *)" \
        --permission-mode acceptEdits \
        --max-turns 250 \
        --verbose \
        2>&1 | tee "logs/run_$(date +%Y%m%d)_iter${i}.log"

    echo "Claude session $i finished."
    # Brief pause so a fast-failing session (rate limit, auth) doesn't burn
    # through all iterations in seconds
    sleep 30
done

echo ""
if "$PYTHON" scripts/verify_outputs.py; then
    echo "✓ All notebooks complete and verified."
    exit 0
fi
echo "✗ Did not converge after $MAX_ITERS iterations. See logs/ and .scratch/verification.json."
exit 1
