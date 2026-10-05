#!/bin/bash
# M9: walker evaluation (B.5) + response measurement (B.6) + demo video, inside the container.
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
PYTHONUNBUFFERED=1 timeout 5400 /isaac-sim/python.sh -u k1_walker/scripts/eval_walker.py --policy k1_walker.pt --headless --video --out walker_eval 2>&1 \
  | tee walker_eval_full.log | grep --line-buffered -vE "Extensions config|^\s*$|\[Warning\]|carb.launcher|interpreter =|read(Stdout|Stderr)|onRead"
mkdir -p walker_eval && cp walker_eval_full.log walker_eval/
tar -czf walker_eval.tar.gz walker_eval
