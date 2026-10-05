#!/usr/bin/env bash
# Token-free: submit the showcase video jobs, wait for them (+ any extra job ids given), fetch results.
#   cluster/showcase.sh [extra job ids...]
source "$(dirname "$0")/common.sh"
state() { rsh "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo sshfail; }
JOBS="$*"
sub() { JOBS="$JOBS $(rsh "cd ~/k1-follow/cluster && condor_submit $* | grep -oE 'cluster [0-9]+' | grep -oE '[0-9]+'")"; }
sub MODE=showcase SCEN=T6 SEED0=1000 N=1 CROWD=6 IGS=none submit_isaac_eval.sub
for spec in "T6 839962" "T8 839962" "T4 839994" "T5 840025"; do set -- $spec; sub MODE=showcase SCEN=$1 SEED0=1000 N=1 CROWD=0 IGS=$2 submit_isaac_eval.sub; done
echo "$(date) showcase submitted:$JOBS" >> "$REPO_ROOT/results/cluster/auto_pipeline.log"
while true; do left=0; for j in $JOBS; do s=$(state $j); [ "$s" = sshfail ] && { echo "STOP: ssh lost (jobs:$JOBS)"; exit 2; }
  { [ "$s" = 1 ] || [ "$s" = 2 ]; } && left=$((left+1)); [ "$s" = 5 ] && { echo "STOP: job $j held (jobs:$JOBS)"; exit 1; }; done
  [ $left = 0 ] && break; sleep 600; done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: jobs$JOBS finished and fetched"
