#!/usr/bin/env bash
# Token-free: wait for the closed-loop debug job; if the K1 stayed up, submit the showcase video jobs, wait, fetch.
source "$(dirname "$0")/common.sh"
DBG=$1; POLL=300
state() { rsh "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo sshfail; }
while true; do s=$(state $DBG); [ "$s" = sshfail ] && { echo "STOP: ssh lost"; exit 2; }; [ "$s" = 1 ] || [ "$s" = 2 ] || break; sleep $POLL; done
out=$(rsh "grep -E 'FELL|Traceback|run_follow\] t=' ~/k1-follow/cluster/jobs/out/debug_follow_$DBG.out | tail -6")
if echo "$out" | grep -qE "FELL|Traceback" || [ -z "$out" ]; then echo "STOP: debug run still failing:"; echo "$out"; exit 1; fi
JOBS=""
sub() { JOBS="$JOBS $(rsh "cd ~/k1-follow/cluster && condor_submit $* | grep -oE 'cluster [0-9]+' | grep -oE '[0-9]+'")"; }
sub MODE=showcase SCEN=T6 SEED0=1000 N=1 CROWD=6 IGS=none submit_isaac_eval.sub
for spec in "T6 839962" "T8 839962" "T4 839994" "T5 840025"; do set -- $spec; sub MODE=showcase SCEN=$1 SEED0=1000 N=1 CROWD=0 IGS=$2 submit_isaac_eval.sub; done
sub submit_walker_eval.sub
echo "$(date) debug OK ($out); submitted:$JOBS" >> "$REPO_ROOT/results/cluster/auto_pipeline.log"
while true; do left=0; for j in $JOBS; do s=$(state $j); [ "$s" = sshfail ] && { echo "STOP: ssh lost (jobs:$JOBS)"; exit 2; }; { [ "$s" = 1 ] || [ "$s" = 2 ]; } && left=$((left+1)); [ "$s" = 5 ] && { echo "STOP: job $j held (jobs:$JOBS)"; exit 1; }; done; [ $left = 0 ] && break; sleep 600; done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: debug OK; showcase + walker eval jobs$JOBS finished and fetched"
