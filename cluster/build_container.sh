#!/bin/bash
# Runs on a CHTC build node (see build_container.sub). Builds the Isaac Lab + K1 image WITHOUT fakeroot
# (definition-file builds need fakeroot, which fails on CHTC build nodes in batch jobs):
#   1. pull nvcr.io/nvidia/isaac-lab:2.3.2 into a user-owned sandbox directory
#   2. modify the sandbox directly (Kit cache symlinks, Booster assets, a few pip packages, env vars)
#   3. pack the sandbox into a .sif
set -euo pipefail
IMAGE="docker://nvcr.io/nvidia/isaac-lab:2.3.2"
BOOSTER_ASSETS_SHA="12a97516f57469078707afb587a5f84c1cb110d1"   # BSD-3-Clause
OUT=isaaclab-2.3.2-k1.sif
SB="$_CONDOR_SCRATCH_DIR/sb"
export APPTAINER_TMPDIR="$_CONDOR_SCRATCH_DIR/apptmp" APPTAINER_CACHEDIR="$_CONDOR_SCRATCH_DIR/aptcache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "[build] host=$(hostname) start=$(date -Is) apptainer=$(apptainer --version)"

echo "[build] 1/3 pulling $IMAGE into sandbox"
apptainer build --sandbox "$SB" "$IMAGE"

echo "[build] 2/3 modifying sandbox"
# Writable Kit locations -> /tmp/kit/* (created per job by run_in_container.sh)
for d in cache data logs; do
  chmod -R u+w "$SB/isaac-sim/kit/$d" 2>/dev/null || true
  rm -rf "$SB/isaac-sim/kit/$d"
  ln -s "/tmp/kit/$d" "$SB/isaac-sim/kit/$d"
done
# Booster assets (pinned)
mkdir -p "$SB/opt"
curl -fsSL "https://github.com/BoosterRobotics/booster_assets/archive/$BOOSTER_ASSETS_SHA.tar.gz" | tar -xz -C "$SB/opt"
mv "$SB/opt/booster_assets-$BOOSTER_ASSETS_SHA" "$SB/opt/booster_assets"
echo "$BOOSTER_ASSETS_SHA" > "$SB/opt/booster_assets/COMMIT"
# Environment
cat > "$SB/.singularity.d/env/90-k1follow.sh" <<'EOF'
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export ISAACLAB_PATH=/workspace/isaaclab
export ISAACSIM_PATH=/isaac-sim
export BOOSTER_ASSETS=/opt/booster_assets
export PYTHONUNBUFFERED=1
EOF
# Python extras into Isaac Sim's python (runs inside the writable, user-owned sandbox)
export HOME="$_CONDOR_SCRATCH_DIR/home"; mkdir -p "$HOME"
apptainer exec --writable --no-home "$SB" /isaac-sim/python.sh -m pip install --no-cache-dir \
  pyyaml pandas imageio imageio-ffmpeg tensorboard \
  || echo "[build] WARN: pip extras failed; continuing"
apptainer exec --writable --no-home "$SB" /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps -e /opt/booster_assets \
  || echo "[build] WARN: booster_assets pip install failed (files still at /opt/booster_assets)"
apptainer exec --no-home "$SB" /isaac-sim/python.sh -m pip freeze > "$SB/opt/pip-freeze.txt" || true

echo "[build] 3/3 packing sif"
apptainer build "$OUT" "$SB"
ls -la "$OUT"
{
  echo "built: $(date -Is) on $(hostname) from $IMAGE"
  echo "booster_assets: $BOOSTER_ASSETS_SHA"
  echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
  echo "size: $(du -h "$OUT" | cut -f1)"
  echo "--- K1 files"; ls "$SB/opt/booster_assets/robots/K1"
  echo "--- pip freeze"; cat "$SB/opt/pip-freeze.txt"
} > build_meta.txt
chmod -R u+w "$SB" 2>/dev/null || true
rm -rf "$SB" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "[build] done $(date -Is)"
