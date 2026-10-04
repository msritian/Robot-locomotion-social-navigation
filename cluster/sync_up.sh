#!/usr/bin/env bash
# Copy code + configs to $REMOTE_WORKDIR on the cluster (results, caches, venvs, git data excluded).
# Safe to rerun: rsync only sends changes; nothing remote outside REMOTE_WORKDIR is touched.
source "$(dirname "$0")/common.sh"
check_connection

log "syncing $REPO_ROOT -> $REMOTE:$REMOTE_WORKDIR"
rsh "mkdir -p '$REMOTE_WORKDIR' '$REMOTE_BIGDATA'/{containers,checkpoints,videos,logs,isaac_cache}"
rsync -az --delete \
  -e "${SSH_CMD[*]}" \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.egg-info/' \
  --exclude '.pytest_cache/' --exclude 'results/' --exclude 'videos/' --exclude 'models/' \
  --exclude 'cluster/cluster.env' --exclude 'logs/' --exclude 'outputs/' \
  --exclude 'cluster/jobs/' \
  "$REPO_ROOT/" "$REMOTE:$REMOTE_WORKDIR/"
log "done. Remote tree:"
rsh "cd '$REMOTE_WORKDIR' && ls"
