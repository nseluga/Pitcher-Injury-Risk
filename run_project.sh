#!/bin/bash
set -e

echo "Starting Claude autonomous run..."

claude -p "
Read claude_instructions.md and follow it completely. 
If there are any obvious and critical changes that you want to make you can besides discarding large chunks of data.
Otherwise, just follow the instructions as they are. 
" \
--allowedTools "Read,Edit,Bash(ls *),Bash(find *),Bash(python *),Bash(python3 *),Bash(pytest *),Bash(python -m pytest *),Bash(jupyter nbconvert *)" \
--permission-mode acceptEdits

echo "Claude process finished."