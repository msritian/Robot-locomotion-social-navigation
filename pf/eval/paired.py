"""Paired comparisons: every method runs on the same (scenario, seed) episodes, so method differences are
estimated per episode pair and bootstrapped (95% CI). Writes results/<ID>/paired_vs_<baseline>.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pf.config import REPO_ROOT

METRICS = ["tracking_rate", "success", "turn_lag", "lost_events", "recovery_rate", "wrong_switches",
           "person_collisions", "personal_space_compliance"]


def paired(exp_id, baseline, n_boot=4000, seed=0):
    d = pd.read_csv(REPO_ROOT / "results" / exp_id / "episodes.csv")
    d = d[d["error"].isna() | (d["error"].astype(str) == "")]
    rng = np.random.default_rng(seed)
    rows = []
    for scope in ["ALL"] + sorted(d.scenario.unique()):
        dd = d if scope == "ALL" else d[d.scenario == scope]
        piv = {m: dd.pivot_table(index=["scenario", "seed"], columns="method", values=m) for m in METRICS if m in dd}
        for method in dd.method.unique():
            if method == baseline:
                continue
            r = {"scope": scope, "method": method, "baseline": baseline}
            for m, p in piv.items():
                if method not in p or baseline not in p:
                    continue
                diff = (p[method] - p[baseline]).dropna().values
                if len(diff) == 0:
                    continue
                bs = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
                r[m] = diff.mean()
                r[m + "_lo"], r[m + "_hi"] = np.percentile(bs, [2.5, 97.5])
                r["n"] = len(diff)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "results" / exp_id / f"paired_vs_{baseline.replace(' ', '_')}.csv", index=False)
    return out


def fmt(out, metric):
    o = out[["scope", "method", metric, metric + "_lo", metric + "_hi"]].copy()
    o["sig"] = np.where((o[metric + "_lo"] > 0) | (o[metric + "_hi"] < 0), "*", "")
    return o.round(3)


if __name__ == "__main__":
    for exp, base in (("E2", "C0"), ("E3", "M2+S1"), ("E4", "delta=0.5")):
        try:
            o = paired(exp, base)
            print(exp, "vs", base)
            print(fmt(o, "tracking_rate").to_string(index=False))
        except FileNotFoundError:
            pass
