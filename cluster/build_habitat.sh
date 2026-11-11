#!/bin/bash
# Runs on a CHTC build node (see build_habitat.sub). Builds the EVT-Bench image (Habitat-Sim 0.3.1 headless +
# OmTrackVLA's habitat-lab fork + torch/transformers for the OmTrackVLA planner + our brain's deps).
# Same no-fakeroot recipe as build_container.sh: pull into a user-owned sandbox, modify it, pack a .sif.
set -euo pipefail
IMAGE="docker://continuumio/miniconda3:24.7.1-0"
OMT_SHA="e9cb1fbd57f8dbcf98c460914251cbcbe20f2f57"
OUT=evt-habitat.sif
SB="$_CONDOR_SCRATCH_DIR/sb"
export APPTAINER_TMPDIR="$_CONDOR_SCRATCH_DIR/apptmp" APPTAINER_CACHEDIR="$_CONDOR_SCRATCH_DIR/aptcache"
export HOME="$_CONDOR_SCRATCH_DIR/home"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$HOME"
echo "[build] host=$(hostname) start=$(date -Is) apptainer=$(apptainer --version)"

echo "[build] 1/3 pulling $IMAGE into sandbox"
apptainer build --sandbox "$SB" "$IMAGE"
X() { apptainer exec --writable --no-home --cleanenv --pwd /tmp "$SB" "$@"; }

echo "[build] 2/3 conda env + pip"
X /opt/conda/bin/conda create -y -p /opt/hab -c conda-forge -c aihabitat \
  python=3.9 cmake=3.14.0 habitat-sim=0.3.1 withbullet headless libglvnd libegl libopengl libgl
P=/opt/hab/bin/python
X $P -m pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
X $P -m pip install --no-cache-dir "transformers==4.57.3" "huggingface_hub<1.0" accelerate safetensors \
  imageio imageio-ffmpeg tqdm pillow scipy pandas pyyaml matplotlib
mkdir -p "$SB/opt/omt"
curl -fsSL "https://github.com/om-ai-lab/OmTrackVLA/archive/$OMT_SHA.tar.gz" | tar -xz --strip-components=1 -C "$SB/opt/omt"
echo "$OMT_SHA" > "$SB/opt/omt/COMMIT"
X $P -m pip install --no-cache-dir -e /opt/omt/habitat-lab
cat > "$SB/.singularity.d/env/90-evt.sh" <<'EOF'
export PATH=/opt/hab/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
EOF
X $P -c "import habitat_sim, habitat, torch, transformers; print('habitat_sim', habitat_sim.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
X $P -m pip freeze > "$_CONDOR_SCRATCH_DIR/pip-freeze.txt" || true

echo "[build] 3/3 packing sif"
apptainer build "$OUT" "$SB"
{
  echo "built: $(date -Is) on $(hostname) from $IMAGE"
  echo "OmTrackVLA: $OMT_SHA"
  echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
  echo "size: $(du -h "$OUT" | cut -f1)"
  echo "--- pip freeze"; cat "$_CONDOR_SCRATCH_DIR/pip-freeze.txt"
} > build_habitat_meta.txt
chmod -R u+w "$SB" 2>/dev/null || true
rm -rf "$SB" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "[build] done $(date -Is)"
