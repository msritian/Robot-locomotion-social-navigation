#!/usr/bin/env bash
# Start an interactive GPU job in the Isaac Lab container. Needs a real terminal (TTY):
# run it yourself in a terminal, not through a non-interactive tool. Inside the job run:
#   source run_in_container.sh true   # sets up HOME, Kit dirs, Vulkan ICD
source "$(dirname "$0")/common.sh"
check_connection
log "requesting an interactive GPU slot (may wait in the queue)"
exec "${SSH_CMD[@]}" -t "$REMOTE" "cd '$REMOTE_WORKDIR/cluster' && condor_submit -i interactive.sub"
