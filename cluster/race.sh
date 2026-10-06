#!/usr/bin/env bash
# Token-free: two copies of a job (CHTC + OSPool). When one starts, remove the other. Exit when the plumbing test
# finishes (either copy), is held, or the connection drops.
source "$(dirname "$0")/common.sh"
st() { rsh "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo ssh; }
PA=6656509; PB=6666950; EA=6656510; EB=6666951; pairs_done=""
while true; do
  for pair in "$PA $PB" "$EA $EB"; do set -- $pair
    a=$(st $1); b=$(st $2); [ "$a" = ssh ] && { echo "STOP: ssh lost"; exit 2; }
    [ "$a" = 2 ] && [ "$b" = 1 ] && rsh "condor_rm $2" >/dev/null
    [ "$b" = 2 ] && [ "$a" = 1 ] && rsh "condor_rm $1" >/dev/null
  done
  p1=$(st $PA); p2=$(st $PB)
  { [ "$p1" = 5 ] || [ "$p2" = 5 ]; } && { echo "STOP: plumbing copy held ($PA:$p1 $PB:$p2)"; exit 1; }
  { [ "$p1" = done ] || [ "$p2" = done ]; } && { [ "$p1" != 2 ] && [ "$p2" != 2 ]; } && break
  sleep 300
done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
echo "DONE: plumbing test finished ($PA:$p1 $PB:$p2); results fetched"
