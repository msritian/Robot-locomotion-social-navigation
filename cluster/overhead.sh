#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
for n in hospital warehouse; do timeout 1800 /isaac-sim/python.sh -u isaac_follow/env_overhead.py --name $n --out overhead --headless 2>&1 | grep --line-buffered -E "overhead\\]|Traceback|Error:|File \""; done
tar -czf overhead.tar.gz overhead
