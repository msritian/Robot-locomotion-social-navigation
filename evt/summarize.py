"""EVT-Bench summary (SR / TR / CR, as in the TrackVLA tables) from per-episode result jsons.

    python evt/summarize.py results/evt/out/ours/stt [results/evt/out/omtrackvla/stt ...]
"""
import glob
import json
import os
import sys
from collections import Counter


def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "*", "*.json")):
        if not f.endswith("_info.json"):
            rows[(os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-5])] = json.load(open(f))
    return rows


def main():
    sets = {d: load(d) for d in sys.argv[1:]}
    common = set.intersection(*(set(r) for r in sets.values())) if len(sets) > 1 else None
    for d, rows in sets.items():
        r = [rows[k] for k in (common if common is not None else rows)]
        n = len(r)
        if n == 0:
            print(f"{d}: no episodes")
            continue
        sr = 100 * sum(bool(x["success"]) for x in r) / n
        tr = 100 * sum(x["following_rate"] for x in r) / n
        cr = 100 * sum(bool(x["collision"]) for x in r) / n
        st = Counter(x["status"] for x in r)
        extra = ""
        if "brain_wrong_switches" in r[0]:
            extra = f"  wrong-person switches/ep {sum(x['brain_wrong_switches'] for x in r) / n:.2f}"
        print(f"{d}: n={n}{' (common)' if common is not None else ''}  SR {sr:.1f}  TR {tr:.1f}  CR {cr:.1f}  "
              f"status {dict(st)}{extra}")


if __name__ == "__main__":
    main()
