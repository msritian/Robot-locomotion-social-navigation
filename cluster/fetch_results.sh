#!/usr/bin/env bash
# Bring results back: job logs/outputs from REMOTE_WORKDIR/cluster/jobs/out, and selected big-data
# folders (logs, videos, exported models) from REMOTE_BIGDATA. Checkpoints stay on the cluster
# unless --checkpoints is passed. Safe to rerun (rsync, never deletes local files).
#   cluster/fetch_results.sh [--checkpoints]
source "$(dirname "$0")/common.sh"
check_connection

mkdir -p "$REPO_ROOT/results/cluster" "$REPO_ROOT/videos" "$REPO_ROOT/models"
RS=(rsync -az -e "${SSH_CMD[*]}")

log "job outputs -> results/cluster/jobs"
"${RS[@]}" "$REMOTE:$REMOTE_WORKDIR/cluster/jobs/out/" "$REPO_ROOT/results/cluster/jobs/" || log "(no job outputs yet)"
log "big-data logs -> results/cluster/logs"
"${RS[@]}" --exclude '*.pt' --exclude '*.pth' "$REMOTE:$REMOTE_BIGDATA/logs/" "$REPO_ROOT/results/cluster/logs/" || true
log "videos -> videos/"
"${RS[@]}" "$REMOTE:$REMOTE_BIGDATA/videos/" "$REPO_ROOT/videos/" || true
log "exported models -> models/"
"${RS[@]}" --include '*/' --include 'k1_walker*' --include '*.onnx' --include '*.yaml' --exclude '*' \
  "$REMOTE:$REMOTE_BIGDATA/checkpoints/export/" "$REPO_ROOT/models/" || true
if [[ "${1:-}" == "--checkpoints" ]]; then
  log "checkpoints -> results/cluster/checkpoints"
  "${RS[@]}" "$REMOTE:$REMOTE_BIGDATA/checkpoints/" "$REPO_ROOT/results/cluster/checkpoints/"
fi
log "done"
