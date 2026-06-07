#!/bin/bash
set -e

claude -p "
Read claude_instructions.md and follow it completely.
Stop after Notebook 9. Do not continue to Notebook 10 or later.
" \
--allowedTools "Read,Edit,Bash(ls *),Bash(find *),Bash(python *),Bash(python3 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter nbconvert *)" \
--permission-mode acceptEdits