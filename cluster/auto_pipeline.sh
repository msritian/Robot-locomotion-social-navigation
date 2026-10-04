#!/usr/bin/env bash
# Local, token-free automation of the post-training chain (Section 20 order). Prints ONE line and exits only when
# a human/agent is needed: all results fetched, or something failed / was held / the SSH connection dropped.
#   cluster/auto_pipeline.sh <train_job_id> <plumb_job_id>
source "$(dirname "$0")/common.sh"
TRAIN=$1; PLUMB=$2; POLL=600
q() { rsh "$@" 2>/dev/null || { echo "STOP: ssh connection lost (reopen the ControlMaster connection)"; exit 2; }; }
state() { q "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo \"done:\$(condor_history $1 -limit 1 -af ExitCode | head -1)\""; }
tb() { q "grep -c Traceback ~/k1-follow/cluster/jobs/out/$1 2>/dev/null || echo 0" | tail -1; }
plumb_done=0
# only NEW tracebacks count (logs keep output from earlier attempts)
TB_TRAIN0=$(tb train_walker_$TRAIN.out); TB_PLUMB0=$(tb isaac_plumb_$PLUMB.out)
# ---- 1. wait for training (and keep an eye on the plumbing test)
while true; do
  t=$(state $TRAIN)
  if [ $plumb_done = 0 ]; then
    p=$(state $PLUMB)
    case "$p" in
      1|2) [ "$(tb isaac_plumb_$PLUMB.out)" != "$TB_PLUMB0" ] && { echo "STOP: plumbing test traceback (job $PLUMB)"; exit 1; } ;;
      done:0) plumb_done=1 ;;
      *) echo "STOP: plumbing test ended badly: $p (job $PLUMB)"; exit 1 ;;
    esac
  fi
  case "$t" in
    1|2) [ "$(tb train_walker_$TRAIN.out)" != "$TB_TRAIN0" ] && { echo "STOP: training traceback (job $TRAIN)"; exit 1; } ;;
    done:0) break ;;
    *) echo "STOP: training ended badly: $t (job $TRAIN)"; exit 1 ;;
  esac
  sleep $POLL
done
# ---- 2. export -> submit walker eval + showcase jobs
q "cd ~/k1-follow/cluster && tar -xzf jobs/out/walker_export_$TRAIN.tar.gz -C jobs/ out/k1_walker.pt && mv jobs/out/k1_walker.pt jobs/k1_walker.pt && ls -la jobs/k1_walker.pt" >/dev/null \
  || { echo "STOP: could not extract exported walker from walker_export_$TRAIN.tar.gz"; exit 1; }
submit() { q "cd ~/k1-follow/cluster && condor_submit $* | grep -oE 'cluster [0-9]+' | grep -oE '[0-9]+'"; }
JOBS="$(submit submit_walker_eval.sub)"
JOBS="$JOBS $(submit MODE=showcase SCEN=T6 SEED0=1000 N=1 CROWD=6 IGS=none submit_isaac_eval.sub)"
for spec in "T6 839962" "T8 839962" "T4 839994" "T5 840025"; do
  set -- $spec
  JOBS="$JOBS $(submit MODE=showcase SCEN=$1 SEED0=1000 N=1 CROWD=0 IGS=$2 submit_isaac_eval.sub)"
done
echo "$(date) submitted: $JOBS" >> "$REPO_ROOT/results/cluster/auto_pipeline.log"
# ---- 3. wait for all of them
while true; do
  left=0
  for j in $JOBS; do
    s=$(state $j)
    case "$s" in 1|2) left=$((left+1));; done:0) ;; *) bad="$bad $j:$s";; esac
  done
  [ $left = 0 ] && break
  sleep $POLL
done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: training $TRAIN finished; jobs $JOBS finished${bad:+ (problems:$bad)}; results fetched to results/cluster/jobs"
