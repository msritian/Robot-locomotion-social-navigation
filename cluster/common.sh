#!/usr/bin/env bash
# Shared helpers for the local-side cluster scripts. Source this file; do not run it.
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CLUSTER_DIR/.." && pwd)"

if [[ ! -f "$CLUSTER_DIR/cluster.env" ]]; then
  echo "ERROR: $CLUSTER_DIR/cluster.env missing. Copy cluster.env.example and fill it in." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CLUSTER_DIR/cluster.env"

for v in CLUSTER_HOST CLUSTER_USER REMOTE_WORKDIR REMOTE_BIGDATA; do
  if [[ -z "${!v:-}" ]]; then
    echo "ERROR: $v is empty in cluster/cluster.env" >&2
    exit 1
  fi
done

REMOTE="$CLUSTER_USER@$CLUSTER_HOST"
SSH_CMD=(ssh -o "ControlPath=${SSH_CONTROL_PATH:-$HOME/.ssh/cm/%r@%h:%p}" -o BatchMode=yes ${SSH_OPTS:-})

check_connection() {
  if ! "${SSH_CMD[@]}" "$REMOTE" true 2>/dev/null; then
    cat >&2 <<EOF
ERROR: no open SSH connection to $REMOTE.
Open one in your own terminal (password + Duo), then rerun:
  mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
  ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm/%r@%h:%p -o ControlPersist=12h -fN $REMOTE
EOF
    exit 1
  fi
}

rsh() { "${SSH_CMD[@]}" "$REMOTE" "$@"; }

log() { echo "[$(date +%H:%M:%S)] $*"; }
