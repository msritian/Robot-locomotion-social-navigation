#!/usr/bin/env bash
# Token-free: wait for one showcase job, fetch it, copy its videos to videos/photoreal, print its metrics.
source "$(dirname "$0")/common.sh"
J=$1
while true; do s=$(rsh "s=\$(condor_q $J -af JobStatus); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo ssh)
  [ "$s" = ssh ] && { echo "STOP: ssh lost"; exit 2; }; [ "$s" = done ] && break; [ "$s" = 5 ] && { echo "STOP: $J held"; exit 1; }
  sleep 600; done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
d="$REPO_ROOT/results/cluster/jobs/final_$J"; mkdir -p "$d" "$REPO_ROOT/videos/photoreal"
tar -xzf "$REPO_ROOT/results/cluster/jobs/d_out_showcase_$J.tar.gz" -C "$d" && cp "$d"/d_out/showcase_*_C0.mp4 "$d"/d_out/showcase_*_full.mp4 "$REPO_ROOT/videos/photoreal/" 2>/dev/null
for f in "$d"/d_out/metrics_*.json; do python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split('/')[-1], 'fell=%s track=%.2f wrong=%s pcoll=%s lost=%s rec=%s'%(d['fell'],d['tracking_rate'],d['wrong_switches'],d['person_collisions'],d['lost_events'],d['recovery_rate']))" "$f"; done
echo "DONE $J"
