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
      LOG=d_out/full_${sc}_${scene}_${s}_${m}.log
      PYTHONUNBUFFERED=1 setsid /isaac-sim/python.sh -u isaac_follow/run_follow.py --headless --scenario "$sc" --seed "$s" --method "$m" \
          --policy "$POLICY" --out d_out $LIM "${extra[@]}" "${igs[@]}" >> "$LOG" 2>&1 &
      PID=$!; T0=$(date +%s); NOGPU=0
      # watchdog: kill the whole process group on a hard time limit, or when the node has no usable renderer
      # ("Graphics plugins not available" -> Kit hangs forever on those nodes)
      while kill -0 $PID 2>/dev/null; do sleep 20
        if grep -q "Graphics plugins not available" "$LOG"; then echo "[isaac_eval] no renderer on $(hostname)"; NOGPU=1; kill -9 -$PID; break; fi
        [ $(( $(date +%s) - T0 )) -gt 5400 ] && { echo "[isaac_eval] time limit"; kill -9 -$PID; break; }
      done; wait $PID 2>/dev/null
      grep -E "^\[(follow|result|isaac)|Traceback|Error:" "$LOG" | grep -v "omni" | tail -5
      [ $NOGPU = 1 ] && exit 1     # HTCondor retries on another machine
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
