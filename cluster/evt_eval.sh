#!/bin/bash
# EVT-Bench chunk inside evt-habitat.sif (see submit_evt.sub).
#   evt_eval.sh <agent ours|omtrackvla|probe> <task stt|dt|at> <split_num> <split_id> <every> [extra run_evt.py args]
set -uo pipefail
AGENT=$1 TASK=$2 SPLITN=$3 SPLITI=$4 EVERY=$5; shift 5
S="${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME="$S/home" HF_HOME="$S/hf" MPLBACKEND=Agg
mkdir -p "$HOME" "$S/code" "$S/w/data" "$S/out"
echo "[evt] host=$(hostname) $(date -Is) agent=$AGENT task=$TASK chunk=$SPLITI/$SPLITN every=$EVERY"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
tar -xzf code.tar.gz -C "$S/code"
for f in /opt/omt/*; do [ "$(basename "$f")" = data ] || ln -s "$f" "$S/w/"; done
for f in /opt/omt/data/*; do ln -s "$f" "$S/w/data/"; done
tar -xf evt_data.tar -C "$S/w/data" && rm -f evt_data.tar
# development scenes (MP3D train split), shipped only for --split train runs
[ -f evt_dev.tar ] && tar -xf evt_dev.tar -C "$S/w/data" && rm -f evt_dev.tar
# EGL vendor file for the NVIDIA driver bound in by --nv (habitat-sim renders headless through EGL)
if [ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$S/10_nvidia.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$S/10_nvidia.json"
fi
# OmTrackVLA weights (hf_models.tar: siglip/ qwen3/ omtrackvla/; dinov3_model.tar: dinov3-vits16/), all local, no network
if [ -f hf_models.tar ]; then
  M="$S/models"; mkdir -p "$M" && tar -xf hf_models.tar -C "$M" && tar -xf dinov3_model.tar -C "$M" && rm -f hf_models.tar dinov3_model.tar
  # HF config points llm_name at the authors' machine; their loader also insists on a pytorch_model.bin file
  sed -i "s#\"llm_name\": \"[^\"]*\"#\"llm_name\": \"$M/qwen3\"#" "$M/omtrackvla/config.json"
  [ -e "$M/omtrackvla/pytorch_model.bin" ] || ln -s model.safetensors "$M/omtrackvla/pytorch_model.bin"
  export HF_MODEL_DIR="$M/omtrackvla" DINOV3_MODEL_PATH="$M/dinov3-vits16" SIGLIP_MODEL_PATH="$M/siglip" HF_HUB_OFFLINE=1
fi
cd "$S/w"
export PYTHONPATH="$S/code:$S/w/habitat-lab:$S/w"
if [ "$AGENT" = probe ]; then ARGS=(--probe); else ARGS=(--agent "$AGENT"); fi
python "$S/code/evt/run_evt.py" "${ARGS[@]}" --task "$TASK" --split-num "$SPLITN" --split-id "$SPLITI" --every "$EVERY" \
  --save-path "$S/out/$AGENT/$TASK" "$@" 2>&1 | grep -vE "^(Followed|Lost)$|^Planner action|^\(384, 384, [34]\)$"
rc=${PIPESTATUS[0]}
cd "$S" && tar -czf evt_out.tar.gz out
echo "[evt] done rc=$rc $(date -Is)"
exit $rc
