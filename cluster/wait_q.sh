#!/usr/bin/env bash
# Token-free: wait until the given cluster ids leave the queue (no fetch). Exits 1 if any is held, 2 if ssh drops.
source "$(dirname "$0")/common.sh"
while true; do s=$(rsh "condor_q $* -af JobStatus | sort -u | xargs echo" 2>/dev/null) || { echo "STOP: ssh lost"; exit 2; }
  case " $s " in *" 5 "*) echo "STOP: held ($s)"; exit 1;; esac
  if [ -z "${s// /}" ]; then empty=$((${empty:-0}+1)); [ $empty -ge 2 ] && break; sleep 20; continue; fi
  empty=0; sleep ${WAIT_S:-120}; done
echo "DONE: $*"
