#!/bin/bash
# Stage D job body (inside the container): closed-loop K1 following in Isaac Sim.
#   isaac_eval.sh <policy.pt|zero> <mode: quant|showcase|plumb> <scenarios (comma)> <seed0> <n_seeds> [crowd] [interiorgs ids (comma) or none] [seconds]
set -uo pipefail
POLICY=$1; MODE=$2; SCEN=$3; SEED0=$4; N=$5; CROWD=${6:-0}; IGS=${7:-none}; SECS=${8:-0}; METHODS=${9:-C0,full}
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz
[ -f interiorgs_v1.tar ] && tar -xf interiorgs_v1.tar && rm -f interiorgs_v1.tar
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
mkdir -p d_out
LIM=""; [ -f configs/walker_response.yaml ] && LIM="--limits configs/walker_response.yaml"
scenes=(none); [ "$IGS" != none ] && scenes=(${IGS//,/ })
for sc in ${SCEN//,/ }; do
 for scene in "${scenes[@]}"; do
  for s in $(seq "$SEED0" $((SEED0 + N - 1))); do
    for m in ${METHODS//,/ }; do
      extra=(--seconds 60)
      [ "$MODE" = showcase ] && extra=(--video --seconds 75 --crowd "$CROWD")
      [ "$MODE" = plumb ] && extra=(--video --seconds 10 --crowd "$CROWD")
      [ "$SECS" != 0 ] && extra+=(--seconds "$SECS")
      igs=()
      case "$scene" in none) ;; env:*) igs=(--isaac_env "${scene#env:}" --env_maps isaac_follow/env_maps) ;;
                       *) igs=(--interiorgs "$scene" --scene_dir interiorgs) ;; esac
      echo "=== $sc scene=$scene seed $s $m $(date -Is)"
      for attempt in 1 2 3; do   # Kit sometimes segfaults at start-up on some nodes: retry
      PYTHONUNBUFFERED=1 timeout 5400 /isaac-sim/python.sh -u isaac_follow/run_follow.py --headless --scenario "$sc" --seed "$s" --method "$m" \
          --policy "$POLICY" --out d_out $LIM "${extra[@]}" "${igs[@]}" 2>&1 | tee -a d_out/full_${sc}_${scene}_${s}_${m}.log \
          | grep --line-buffered -vE "Extensions config|^\s*$|\[Warning\]|carb.launcher|interpreter =|read(Stdout|Stderr)|onRead"
      tail -5 d_out/full_${sc}_${scene}_${s}_${m}.log | grep -q "Segmentation fault" || break
      echo "[isaac_eval] start-up crash, retry $attempt"; sleep 20
      done
      [ "$MODE" = plumb ] && break
    done
  done
 done
done
tar -czf d_out.tar.gz d_out
# showcase without a video = failed run -> non-zero exit so HTCondor retries on another machine
[ "$MODE" = showcase ] && ! ls d_out/*.mp4 >/dev/null 2>&1 && exit 1
exit 0
