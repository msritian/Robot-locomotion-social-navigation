#!/bin/bash
# M0 hello-world job (Spec A.5): nvidia-smi, driver/Vulkan/internet probes, a few iterations of a
# built-in Isaac Lab locomotion task headless, and a tiny camera-render check. Never exits early:
# every probe is logged so one failure does not hide the others. Results -> hello_out.tar.gz
set -uo pipefail
OUT="$_CONDOR_SCRATCH_DIR/hello_out"
mkdir -p "$OUT"
exec > >(tee "$OUT/hello.log") 2>&1
PY=/isaac-sim/python.sh
LAB=/workspace/isaaclab

step() { echo; echo "===== $* ====="; }

step "nvidia-smi"; nvidia-smi || echo "FAIL nvidia-smi"
step "driver libs bound into the container"
ls /.singularity.d/libs 2>/dev/null | grep -Ei "GLX_nvidia|vulkan|rtcore|gpucomp|nvoptix|EGL_nvidia|cuda" || echo "(no /.singularity.d/libs listing)"
ls /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>&1
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
step "internet"
curl -sS -o /dev/null -w "http %{http_code} in %{time_total}s\n" --max-time 20 https://omniverse-content-production.s3-us-west-2.amazonaws.com/ || echo "FAIL no internet to Omniverse S3"
curl -sS -o /dev/null -w "github %{http_code}\n" --max-time 20 https://github.com || echo "FAIL no github"
step "K1 assets"; ls /opt/booster_assets/robots/K1 || echo "FAIL no K1 assets"

step "Isaac Lab headless train: Isaac-Velocity-Flat-H1-v0, 512 envs, 5 iterations"
cd "$_CONDOR_SCRATCH_DIR"
t0=$(date +%s)
timeout 2400 $PY $LAB/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 --num_envs 512 --max_iterations 5 --headless
echo "train exit=$? seconds=$(( $(date +%s) - t0 ))"
cp -r logs "$OUT/isaaclab_logs" 2>/dev/null || true

step "camera render check (RTX renderer, 1 short video)"
t0=$(date +%s)
timeout 1800 $PY $LAB/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-H1-v0 --num_envs 16 --max_iterations 1 --headless --video --video_length 60 --video_interval 1
echo "render exit=$? seconds=$(( $(date +%s) - t0 ))"
find logs -name "*.mp4" -exec cp {} "$OUT/" \; 2>/dev/null
ls -la "$OUT"

step "kit logs (tail)"
for f in /tmp/kit/logs/*/*/*.log "$HOME"/.nvidia-omniverse/logs/Kit/*/*/*.log; do [ -f "$f" ] && { echo "--- $f"; grep -Ei "error|vulkan|gpu" "$f" | tail -40; }; done
cd "$_CONDOR_SCRATCH_DIR" && tar -czf hello_out.tar.gz hello_out
echo "done"
