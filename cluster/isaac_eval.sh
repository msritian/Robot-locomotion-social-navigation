#!/bin/bash
# Stage D job body (inside the container): closed-loop K1 following in Isaac Sim.
#   isaac_eval.sh <policy.pt|zero> <mode: quant|showcase|plumb> <scenarios (comma)> <seed0> <n_seeds> [crowd] [interiorgs ids (comma) or none] [seconds]
set -uo pipefail
POLICY=$1; MODE=$2; SCEN=$3; SEED0=$4; N=$5; CROWD=${6:-0}; IGS=${7:-none}; SECS=${8:-0}
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
[ -f interiorgs_v1.tar ] && tar -xf interiorgs_v1.tar && rm -f interiorgs_v1.tar
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
mkdir -p d_out
LIM=""; [ -f configs/walker_response.yaml ] && LIM="--limits configs/walker_response.yaml"
scenes=(none); [ "$IGS" != none ] && scenes=(${IGS//,/ })
for sc in ${SCEN//,/ }; do
 for scene in "${scenes[@]}"; do
  for s in $(seq "$SEED0" $((SEED0 + N - 1))); do
    for m in C0 full; do
      extra=(--seconds 60)
      [ "$MODE" = showcase ] && extra=(--video --seconds 75 --crowd "$CROWD")
      [ "$MODE" = plumb ] && extra=(--video --seconds 10 --crowd "$CROWD")
      [ "$SECS" != 0 ] && extra+=(--seconds "$SECS")
      igs=(); [ "$scene" != none ] && igs=(--interiorgs "$scene" --scene_dir interiorgs)
      echo "=== $sc scene=$scene seed $s $m $(date -Is)"
      PYTHONUNBUFFERED=1 /isaac-sim/python.sh -u isaac_follow/run_follow.py --headless --scenario "$sc" --seed "$s" --method "$m" \
          --policy "$POLICY" --out d_out $LIM "${extra[@]}" "${igs[@]}" 2>&1 | tee -a d_out/full_${sc}_${scene}_${s}_${m}.log \
          | grep --line-buffered -vE "Extensions config|^\s*$|\[Warning\]|carb.launcher|interpreter =|read(Stdout|Stderr)|onRead"
      [ "$MODE" = plumb ] && break
    done
  done
 done
done
tar -czf d_out.tar.gz d_out
