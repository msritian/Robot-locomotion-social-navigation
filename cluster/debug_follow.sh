#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
mkdir -p d_out
/isaac-sim/python.sh -u isaac_follow/run_follow.py --headless --scenario T7 --seed 1000 --method C0 --policy k1_walker.pt --seconds 8 --debug --out d_out 2>&1 | grep --line-buffered -E "debug|run_follow|Traceback|Error:|File \""
