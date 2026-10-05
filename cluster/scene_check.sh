#!/bin/bash
set -uo pipefail
cd "$_CONDOR_SCRATCH_DIR" && tar -xzf code.tar.gz && tar -xf interiorgs_v1.tar && rm -f interiorgs_v1.tar
export PYTHONPATH="$PWD:$PWD/k1_walker:${PYTHONPATH:-}"
/isaac-sim/python.sh -u isaac_follow/render_scene_check.py --interiorgs 840025 --scene_dir interiorgs --out scene_check --headless 2>&1 | grep --line-buffered -E "\[check\]|Traceback|Error:|File \"|nurec|NuRec"
tar -czf scene_check.tar.gz scene_check
