#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
for v in A B C; do /isaac-sim/python.sh -u k1_walker/scripts/debug_bisect.py --variant $v --policy k1_walker.pt --headless 2>&1 | grep --line-buffered -E "^\[[ABC]\]|Traceback|Error:|File \""; done
