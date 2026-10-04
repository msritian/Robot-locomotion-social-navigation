#!/bin/bash
# Stage D job body (inside the container): closed-loop K1 following in Isaac Sim.
#   isaac_eval.sh <policy.pt> <mode: quant|showcase> <scenarios (comma)> <seed0> <n_seeds> [crowd]
# quant: C0 and full on n_seeds per scenario, metrics only. showcase: C0 and full, 3-view videos.
set -uo pipefail
POLICY=$1; MODE=$2; SCEN=$3; SEED0=$4; N=$5; CROWD=${6:-0}
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
/isaac-sim/python.sh -m pip install --quiet --no-deps -e . >/dev/null 2>&1 || true
mkdir -p d_out
LIM=""; [ -f configs/walker_response.yaml ] && LIM="--limits configs/walker_response.yaml"
for sc in ${SCEN//,/ }; do
  for s in $(seq "$SEED0" $((SEED0 + N - 1))); do
    for m in C0 full; do
      extra=()
      if [ "$MODE" = showcase ]; then extra=(--video --seconds 75 --crowd "$CROWD"); else extra=(--seconds 60); fi
      echo "=== $sc seed $s $m $(date -Is)"
      /isaac-sim/python.sh isaac_follow/run_follow.py --headless --scenario "$sc" --seed "$s" --method "$m" \
          --policy "$POLICY" --out d_out $LIM "${extra[@]}" 2>&1 | grep -E "run_follow|Traceback|Error|error|\"tracking_rate\"|videos:|FELL" 
    done
  done
done
tar -czf d_out.tar.gz d_out
