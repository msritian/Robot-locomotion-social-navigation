#!/usr/bin/env bash
# Token-free: wait until all given jobs leave the queue (or one is held / ssh drops), then fetch results.
source "$(dirname "$0")/common.sh"
while true; do left=0
  for j in "$@"; do s=$(rsh "s=\$(condor_q $j -af JobStatus); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo ssh)
    [ "$s" = ssh ] && { echo "STOP: ssh lost"; exit 2; }; [ "$s" = 5 ] && { echo "STOP: $j held"; exit 1; }
    [ "$s" = done ] || left=$((left+1)); done
  [ $left = 0 ] && break; sleep 300; done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: $*"
