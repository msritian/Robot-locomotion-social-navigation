#!/bin/bash
# Download SAGE-3D InteriorGS scenes (splat USDZ + collision USD; CC-BY-NC-4.0, ungated) -> interiorgs.tar
set -euo pipefail
mkdir -p interiorgs
for id in "$@"; do
  echo "[fetch] $id $(date -Is)"
  curl -fsSL --retry 3 -o interiorgs/$id.usdz \
    "https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz/resolve/main/InteriorGS_usdz/$id.usdz"
  curl -fsSL --retry 3 -o interiorgs/${id}_collision.usd \
    "https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh/resolve/main/Collision_Mesh/$id/${id}_collision.usd"
done
ls -la interiorgs
tar -cf interiorgs.tar interiorgs
