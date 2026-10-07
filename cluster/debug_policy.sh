#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz && mkdir -p wexp && tar -xzf walker_export_6635156.tar.gz -C wexp
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
/isaac-sim/python.sh -u k1_walker/scripts/debug_policy.py --checkpoint wexp/out/model_3998.pt --policy k1_walker.pt --headless 2>&1 | grep --line-buffered -vE "Extensions config|Warning|carb.launcher|interpreter|onRead|readStd|^\s*$"
