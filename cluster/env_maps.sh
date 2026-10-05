#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
for n in office hospital warehouse; do timeout 1500 /isaac-sim/python.sh -u isaac_follow/env_to_map.py --name $n --headless 2>&1 | grep --line-buffered -E "\\[map\\]|Traceback|Error:|File \""; done
tar -czf env_maps.tar.gz env_maps
