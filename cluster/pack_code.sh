#!/usr/bin/env bash
# Sync code to the cluster and pack the parts jobs need (k1_walker, pf, isaac_follow, configs, models)
# into $REMOTE_WORKDIR/cluster/jobs/code.tar.gz. Safe to rerun.
source "$(dirname "$0")/common.sh"
bash "$CLUSTER_DIR/sync_up.sh" >/dev/null
log "syncing model checkpoints needed by jobs (predictors)"
rsync -az -e "${SSH_CMD[*]}" --include 'predictor_*.pt' --exclude '*' "$REPO_ROOT/models/" "$REMOTE:$REMOTE_WORKDIR/models/" || true
log "packing code.tar.gz on the access point"
rsh "cd '$REMOTE_WORKDIR' && mkdir -p cluster/jobs/out && tar -czf cluster/jobs/code.tar.gz k1_walker pf isaac_follow configs models pyproject.toml && ls -la cluster/jobs/code.tar.gz"
