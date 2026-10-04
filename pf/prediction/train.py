"""Train P1/P2 and run E1 (Section 7.6 / 11): ADE/FDE at 1 s and 2 s, minADE/minFDE over 3 modes,
overall and on the turn subset (true heading change > 45 deg within the next 2 s), mean +- 95% bootstrap CI.

    python -m pf.prediction.train --variant P1
    python -m pf.prediction.train --variant P2
    python -m pf.prediction.train --eval        # E1 table on the held-out test set -> results/E1/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pf.config import REPO_ROOT
from pf.prediction.base import ConstantVelocity
from pf.prediction.data import OUT_DIR
from pf.prediction.model import VARIANT_FEATURES, LearnedPredictor, PredNet, features, wta_loss

MODELS = REPO_ROOT / "models"


def load(split):
    d = np.load(OUT_DIR / f"{split}.npz")
    return {k: d[k] for k in d.files}


def to_local_targets(d, variant):
    x, origin, ang = features(d, variant)
    from pf.prediction.model import rotate
    y = rotate(d["fut"] - origin[:, None], ang).astype(np.float32)
    return x, y


def train(variant, epochs=20, batch=512, lr=1e-3, seed=0, hidden=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = load("train"), load("val")
    xtr, ytr = to_local_targets(tr, variant)
    xva, yva = to_local_targets(va, variant)
    net = PredNet(VARIANT_FEATURES[variant], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr)
    Xva, Yva = torch.from_numpy(xva), torch.from_numpy(yva)
    best, best_state, log = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            traj, logits, ls = net(Xtr[j])
            loss, _ = wta_loss(traj, logits, ls, Ytr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        net.eval()
        with torch.no_grad():
            traj, logits, _ = net(Xva)
            pbest = logits.argmax(1)
            err = torch.linalg.norm(traj[torch.arange(len(Yva)), pbest] - Yva, dim=-1)
            val_ade = err.mean().item()
            val_min = torch.linalg.norm(traj - Yva[:, None], dim=-1).mean(-1).min(1).values.mean().item()
        log.append({"epoch": ep, "train_loss": tot / len(perm), "val_ADE2": val_ade, "val_minADE2": val_min})
        print(f"{variant} ep {ep:2d} loss {tot / len(perm):.4f} val ADE@2s {val_ade:.4f} minADE {val_min:.4f}", flush=True)
        if val_ade < best:
            best, best_state = val_ade, {k: v.clone() for k, v in net.state_dict().items()}
    MODELS.mkdir(exist_ok=True)
    path = MODELS / f"predictor_{variant}.pt"
    torch.save({"variant": variant, "hidden": hidden, "state_dict": best_state, "val_ADE2": best,
                "train_samples": len(Xtr), "val_samples": len(Xva), "epochs": epochs, "seed": seed}, path)
    out = REPO_ROOT / "results" / "M6"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / f"train_log_{variant}.csv", index=False)
    print(f"saved {path} (best val ADE@2s {best:.4f}) in {time.time() - t0:.0f}s")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1) if len(x) < 20000 else \
        np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def predict_all(name, d):
    if name == "P0":
        p0 = ConstantVelocity()
        modes = np.stack([p0.predict({k: d[k][i] for k in ("pos", "vel", "body", "head", "vis")}, 0.0).modes
                          for i in range(len(d["fut"]))])
        probs = np.tile([1.0, 0.0, 0.0], (len(modes), 1))
        return modes, probs
    pred = LearnedPredictor(MODELS / f"predictor_{name}.pt")
    torch.set_num_threads(4)
    out_m, out_p = [], []
    for i in range(0, len(d["fut"]), 8192):
        m, p, _ = pred.predict_batch({k: d[k][i:i + 8192] for k in ("pos", "vel", "body", "head", "vis")})
        out_m.append(m); out_p.append(p)
    return np.concatenate(out_m), np.concatenate(out_p)


def per_sample_errors(modes, probs, fut):
    best = probs.argmax(1)
    idx = np.arange(len(fut))
    e = np.linalg.norm(modes - fut[:, None], axis=-1)       # (N, K, 20)
    eb = e[idx, best]
    return {"ADE@1s": eb[:, :10].mean(1), "FDE@1s": eb[:, 9], "ADE@2s": eb.mean(1), "FDE@2s": eb[:, 19],
            "minADE@2s": e.mean(-1).min(1), "minFDE@2s": e[:, :, 19].min(1)}


def evaluate(names=("P0", "P1", "P2"), split="test"):
    d = load(split)
    rows, raw = [], {}
    for name in names:
        if name != "P0" and not (MODELS / f"predictor_{name}.pt").exists():
            continue
        modes, probs = predict_all(name, d)
        errs = per_sample_errors(modes, probs, d["fut"])
        raw[name] = errs
        for subset, mask in (("overall", np.ones(len(d["fut"]), bool)), ("turn", d["turn"].astype(bool))):
            for metric, v in errs.items():
                m, lo, hi = boot_ci(v[mask])
                rows.append({"predictor": name, "subset": subset, "metric": metric, "mean": m, "ci_lo": lo,
                             "ci_hi": hi, "n": int(mask.sum())})
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "E1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"E1_{split}.csv", index=False)
    # error vs horizon (for the report figure)
    hz = []
    for name in raw:
        modes, probs = predict_all(name, d)
        e = np.linalg.norm(modes[np.arange(len(probs)), probs.argmax(1)] - d["fut"], axis=-1)
        for subset, mask in (("overall", np.ones(len(e), bool)), ("turn", d["turn"].astype(bool))):
            for k in range(e.shape[1]):
                hz.append({"predictor": name, "subset": subset, "t": (k + 1) * 0.1, "err": float(e[mask, k].mean())})
    pd.DataFrame(hz).to_csv(out / f"E1_{split}_horizon.csv", index=False)
    with open(out / f"E1_{split}_meta.json", "w") as f:
        json.dump({"split": split, "n_samples": int(len(d["fut"])), "n_turn": int(d["turn"].sum()),
                   "seeds": [int(d["seed"].min()), int(d["seed"].max())]}, f, indent=1)
    piv = df.pivot_table(index=["subset", "metric"], columns="predictor", values="mean").round(3)
    print(piv.to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["P1", "P2"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    if args.variant:
        train(args.variant, args.epochs)
    if args.eval:
        evaluate(split=args.split)


if __name__ == "__main__":
    main()
