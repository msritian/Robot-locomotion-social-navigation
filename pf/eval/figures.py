"""Report figures from results CSVs (static PNGs for REPORT.md).

Style: categorical hues in a fixed order (validated reference palette, slots 1-8), color follows the
method (never its rank), thin marks, recessive grid, legend for >= 2 series, 95% CI error bars.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pf.config import REPO_ROOT  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RES = REPO_ROOT / "results"

BAR_METRICS = [("tracking_rate", "Tracking rate"), ("turn_lag", "Turn lag (s)"),
               ("wrong_switches", "Wrong-person switches / ep"), ("recovery_rate", "Recovery rate"),
               ("personal_space_compliance", "Personal-space compliance")]


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def e1_horizon():
    f = RES / "E1" / "E1_test_horizon.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, subset in zip(axes, ("overall", "turn")):
        for i, (name, g) in enumerate(df[df.subset == subset].groupby("predictor", sort=True)):
            ax.plot(g.t, g.err, color=PALETTE[i], lw=2, label=name)
            ax.annotate(name, (g.t.iloc[-1], g.err.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=TEXT, va="center")
        ax.set_title("All samples" if subset == "overall" else "Turn subset (>45° within 2 s)", fontsize=10, color=TEXT)
        ax.set_xlabel("Prediction horizon (s)", fontsize=9, color=TEXT)
        _style(ax)
    axes[0].set_ylabel("Error of most likely mode (m)", fontsize=9, color=TEXT)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "E1" / "E1_error_vs_horizon.png", dpi=150)
    plt.close(fig)


def experiment_bars(exp_id):
    f = RES / exp_id / "summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    methods = list(dict.fromkeys(df.method))
    scen = [s for s in dict.fromkeys(df.scenario) if s != "ALL"] + ["ALL"]
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    mets = [(m, l) for m, l in BAR_METRICS if m in df and df[m].notna().any()]
    fig, axes = plt.subplots(len(mets), 1, figsize=(max(6, 1.1 * len(scen) * max(1, len(methods) / 3)), 2.3 * len(mets)))
    axes = np.atleast_1d(axes)
    w = 0.8 / len(methods)
    x = np.arange(len(scen))
    for ax, (mtr, label) in zip(axes, mets):
        for i, m in enumerate(methods):
            g = df[df.method == m].set_index("scenario").reindex(scen)
            y = g[mtr].values
            err = np.vstack([y - g[mtr + "_lo"].values, g[mtr + "_hi"].values - y])
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, y, w * 0.92, color=color[m], label=m,
                   yerr=np.nan_to_num(err), error_kw=dict(lw=0.8, ecolor=MUTED, capsize=0))
        ax.set_xticks(x, scen)
        ax.set_ylabel(label, fontsize=8, color=TEXT)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=min(4, len(methods)), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle(f"{exp_id}: mean ± 95% bootstrap CI over episodes", fontsize=10, color=TEXT, y=1.0)
    fig.tight_layout()
    fig.savefig(RES / exp_id / f"{exp_id}_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all(ids=("E1", "E2", "E3", "E4", "E5", "E6")):
    if "E1" in ids:
        e1_horizon()
    for e in ids:
        if e != "E1":
            experiment_bars(e)


if __name__ == "__main__":
    make_all()
