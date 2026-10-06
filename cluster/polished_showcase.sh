#!/usr/bin/env bash
# Token-free: submit the photoreal showcase set (flocking-capable: CHTC or OSPool), wait, fetch, copy videos.
source "$(dirname "$0")/common.sh"
st() { rsh "s=\$(condor_q $1 -af JobStatus | head -1); [ -n \"\$s\" ] && echo \$s || echo done" 2>/dev/null || echo ssh; }
JOBS=""
for spec in "T6 warehouse" "T8 warehouse" "T4 hospital" "T5 hospital" "T6 office" "T5 office"; do for meth in C0 full; do set -- $spec
  j=$(rsh "cd ~/k1-follow/cluster && condor_submit MODE=showcase SCEN=$1 SEED0=1000 N=1 CROWD=0 IGS=env:$2 SECS=120 METHODS=$meth submit_isaac_eval_ospool.sub | grep -oE 'cluster [0-9]+' | grep -oE '[0-9]+'")
  JOBS="$JOBS $j"; echo "$(date) polished $1 $2 $meth -> $j" >> "$REPO_ROOT/results/cluster/auto_pipeline.log"; done
done
echo "submitted:$JOBS"
while true; do left=0; bad=""
  for j in $JOBS; do s=$(st $j); [ "$s" = ssh ] && { echo "STOP: ssh lost (jobs:$JOBS)"; exit 2; }
    { [ "$s" = 1 ] || [ "$s" = 2 ]; } && left=$((left+1)); [ "$s" = 5 ] && bad="$bad $j"; done
  [ -n "$bad" ] && { echo "STOP: held:$bad (jobs:$JOBS)"; exit 1; }
  [ $left = 0 ] && break; sleep 600; done
bash "$CLUSTER_DIR/fetch_results.sh" >/dev/null 2>&1
mkdir -p "$REPO_ROOT/videos/photoreal"
for j in $JOBS; do f="$REPO_ROOT/results/cluster/jobs/d_out_showcase_$j.tar.gz"; [ -f "$f" ] || continue
  d="$REPO_ROOT/results/cluster/jobs/final_$j"; mkdir -p "$d"; tar -xzf "$f" -C "$d"
  cp "$d"/d_out/showcase_*_{C0,full}.mp4 "$REPO_ROOT/videos/photoreal/" 2>/dev/null
done
echo "DONE: final showcase jobs$JOBS fetched; videos in videos/photoreal"; ls "$REPO_ROOT/videos/photoreal"
