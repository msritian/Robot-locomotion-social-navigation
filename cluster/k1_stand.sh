#!/bin/bash
# M1: K1 stand test inside the container. Output -> stand_out.tar.gz
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD/k1_walker:${PYTHONPATH:-}"
/isaac-sim/python.sh k1_walker/scripts/stand_test.py --headless --video --out stand_out 2>&1 | grep -vE "Extensions config|^\s*$"
echo "stand exit=${PIPESTATUS[0]}"
if [ ! -f stand_out/stand_test.json ]; then
  echo "video run failed; retrying without cameras"
  /isaac-sim/python.sh k1_walker/scripts/stand_test.py --headless --out stand_out 2>&1 | grep -vE "Extensions config|^\s*$"
fi
tar -czf stand_out.tar.gz stand_out
