#!/usr/bin/env bash
# Token-free: two copies of a job (CHTC + OSPool). When one starts, remove the other. Exit when the plumbing test
# finishes (either copy), is held, or the connection drops.
source "$(dirname "$0")/common.sh"
st() { rsh "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo ssh; }
PA=$1; PB=$2; EA=$3; EB=$4; runs=0
while true; do
  for pair in "$PA $PB" "$EA $EB"; do set -- $pair
    a=$(st $1); b=$(st $2); [ "$a" = ssh ] && { echo "STOP: ssh lost"; exit 2; }
    [ "$a" = 2 ] && [ "$b" = 1 ] && rsh "condor_rm $2" >/dev/null
    [ "$b" = 2 ] && [ "$a" = 1 ] && rsh "condor_rm $1" >/dev/null
  done
  p1=$(st $PA); p2=$(st $PB)
  if [ "$p1" = 2 ] || [ "$p2" = 2 ]; then runs=$((runs+1)); j=$PA; [ "$p2" = 2 ] && j=$PB
    prog=$(rsh "grep -c 'run_follow\] t=' ~/k1-follow/cluster/jobs/out/isaac_plumb_$j.out 2>/dev/null")
    [ $runs -ge 9 ] && [ "${prog:-0}" = 0 ] && { echo "STOP: plumbing job $j running 45+ min without sim progress"; exit 1; }
  fi
  { [ "$p1" = 5 ] || [ "$p2" = 5 ]; } && { echo "STOP: plumbing copy held ($PA:$p1 $PB:$p2)"; exit 1; }
  { [ "$p1" = done ] || [ "$p2" = done ]; } && { [ "$p1" != 2 ] && [ "$p2" != 2 ]; } && break
  sleep 300
done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: plumbing test finished ($PA:$p1 $PB:$p2); results fetched"
