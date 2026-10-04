#!/bin/bash
# Common entrypoint INSIDE the Isaac Lab container on a CHTC execute node.
#   run_in_container.sh <command> [args...]
# Sets up writable Kit/home locations in job scratch, makes the NVIDIA Vulkan driver visible if the
# image lacks an ICD manifest, prints a short environment report, then runs the command.
set -euo pipefail
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
export HOME="$SCRATCH/home"
mkdir -p "$HOME" /tmp/kit/cache /tmp/kit/data /tmp/kit/logs
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y PYTHONUNBUFFERED=1

echo "[run_in_container] host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "WARN: nvidia-smi failed"

# Vulkan ICD: Isaac Sim needs the NVIDIA Vulkan driver (libGLX_nvidia.so.0, bound by --nv).
if ! ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  cat > "$SCRATCH/nvidia_icd.json" <<'EOF'
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0" } }
EOF
  export VK_ICD_FILENAMES="$SCRATCH/nvidia_icd.json" VK_DRIVER_FILES="$SCRATCH/nvidia_icd.json"
  echo "[run_in_container] no NVIDIA ICD in image -> using $VK_ICD_FILENAMES"
fi

exec "$@"
