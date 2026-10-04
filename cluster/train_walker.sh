#!/bin/bash
# Stage B training job body (runs inside the Isaac Lab container via run_in_container.sh).
# Self-checkpointing (HTCondor checkpoint_exit_code = 85): trains CHUNK more iterations, then exits 85 so
# HTCondor saves ./logs (transfer_checkpoint_files) and restarts the job, which resumes from the newest
# checkpoint. When TOTAL iterations are reached it exports the policy (TorchScript + ONNX) and exits 0.
#   train_walker.sh <TOTAL_ITERS> <CHUNK_ITERS> <NUM_ENVS>
set -uo pipefail
TOTAL=${1:-4000}
CHUNK=${2:-500}
NUM_ENVS=${3:-4096}
PY=/isaac-sim/python.sh
cd "$_CONDOR_SCRATCH_DIR"
tar -xzf code.tar.gz
export PYTHONPATH="$PWD/k1_walker:${PYTHONPATH:-}"
LOGROOT=logs/rsl_rl/k1_flat
mkdir -p "$LOGROOT" out

latest=$(ls -1 $LOGROOT/*/model_*.pt 2>/dev/null | awk -F'model_|\\.pt' '{print $(NF-1)" "$0}' | sort -n | tail -1)
done_iters=$(echo "$latest" | awk '{print $1}')
ckpt=$(echo "$latest" | awk '{print $2}')
done_iters=${done_iters:-0}
echo "[train_walker] $(date -Is) host=$(hostname) done_iters=$done_iters total=$TOTAL ckpt=${ckpt:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$done_iters" -lt "$((TOTAL - 1))" ]; then
  n=$(( TOTAL - 1 - done_iters < CHUNK ? TOTAL - 1 - done_iters : CHUNK ))
  extra=()
  if [ -n "${ckpt:-}" ]; then
    extra=(--resume --load_run "$(basename "$(dirname "$ckpt")")" --checkpoint "$(basename "$ckpt")")
  fi
  for attempt in 1 2 3; do
    t0=$(date +%s)
    $PY k1_walker/scripts/train.py --task K1-Velocity-Flat-v0 --headless --num_envs "$NUM_ENVS" \
        --max_iterations "$n" --seed 42 "${extra[@]}" 2>&1 | grep -vE "Extensions config|^\s*$|crashreporter-breakpad.plugin\] \[crash\]  "
    rc=${PIPESTATUS[0]}
    dt=$(( $(date +%s) - t0 ))
    echo "[train_walker] chunk exit=$rc seconds=$dt attempt=$attempt"
    # crash during Kit startup (seen once: segfault in XOpenDisplay on one node): retry
    if [ "$rc" -ne 0 ] && [ "$dt" -lt 120 ]; then sleep 20; continue; fi
    break
  done
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  latest_after=$(ls -1 $LOGROOT/*/model_*.pt 2>/dev/null | awk -F'model_|\\.pt' '{print $(NF-1)}' | sort -n | tail -1)
  if [ "${latest_after:-0}" -lt "$((TOTAL - 1))" ]; then
    echo "[train_walker] checkpointing at iteration $latest_after (exit 85 -> HTCondor restarts the job)"
    exit 85
  fi
fi

# ---- final: export the newest checkpoint (TorchScript + ONNX via Isaac Lab play.py) and package results
latest=$(ls -1 $LOGROOT/*/model_*.pt | awk -F'model_|\\.pt' '{print $(NF-1)" "$0}' | sort -n | tail -1 | awk '{print $2}')
echo "[train_walker] exporting $latest"
timeout 1200 $PY k1_walker/scripts/play.py --task K1-Velocity-Flat-Play-v0 --headless --num_envs 4 \
    --checkpoint "$latest" --video_length 1 2>&1 | grep -vE "Extensions config" | tail -20 &
PLAYPID=$!
# play.py runs forever after exporting; stop it once the export exists
for i in $(seq 1 120); do
  [ -f "$(dirname "$latest")/exported/policy.pt" ] && break
  sleep 5
done
kill $PLAYPID 2>/dev/null; pkill -f play.py 2>/dev/null
cp "$(dirname "$latest")/exported/policy.pt" out/k1_walker.pt 2>/dev/null || echo "WARN: export missing"
cp "$(dirname "$latest")/exported/policy.onnx" out/k1_walker.onnx 2>/dev/null || true
cp "$latest" out/
cp -r "$(dirname "$latest")/params" out/params 2>/dev/null || true
tar -czf walker_logs.tar.gz logs
tar -czf walker_export.tar.gz out
echo "[train_walker] done $(date -Is)"
exit 0
