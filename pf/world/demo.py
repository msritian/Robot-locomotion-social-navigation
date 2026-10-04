"""M3 deliverables: map overview figure + one top-down GIF per layout family.

The robot in these GIFs uses a NAIVE ground-truth chase (it reads the true target position).
It is only there to show the world; it is not one of the methods under test.

    python -m pf.world.demo [--out results/M3] [--seconds 40]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pf.config import REPO_ROOT, load_config  # noqa: E402
from pf.world.maps import FAMILIES, generate_map  # noqa: E402
from pf.world.render import OTHER_C, TARGET_C, TopDownRenderer, save_gif  # noqa: E402
from pf.world.robot import wrap  # noqa: E402
from pf.world.sim import World  # noqa: E402


def naive_chase(world):
    """Demo-only controller using ground truth. NOT a method under test."""
    x, y, yaw = world.robot.pose
    tp = world.people.pos[world.target_index]
    d = np.hypot(tp[0] - x, tp[1] - y)
    err = wrap(np.arctan2(tp[1] - y, tp[0] - x) - yaw)
    vx = np.clip(0.8 * (d - 1.5), -0.2, 0.5) if abs(err) < 0.7 else 0.0
    return np.array([vx, 0.0, np.clip(2.0 * err, -1, 1)])


def map_overview(cfg, out: Path):
    seeds = [0, 1, 1000, 1001]
    fig, axes = plt.subplots(len(FAMILIES), len(seeds), figsize=(4 * len(seeds), 3.4 * len(FAMILIES)))
    meta = []
    for r, fam in enumerate(FAMILIES):
        for c, seed in enumerate(seeds):
            m = generate_map(fam, seed, cfg)
            ax = axes[r, c]
            ax.imshow(m.grid.occ, origin="lower", extent=(0, m.width, 0, m.height), cmap="Greys", vmin=0, vmax=1.4)
            ax.scatter(m.pois[:, 0], m.pois[:, 1], s=10, c="#2ca02c")
            s = m.summary()
            ax.set_title(f"{fam} seed {seed} ({s['split']})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            meta.append(s)
    fig.tight_layout()
    fig.savefig(out / "maps_overview.png", dpi=110)
    plt.close(fig)
    return meta


def family_gif(cfg, fam, seed, out: Path, seconds: float):
    w = World(cfg).reset(seed, fam, n_others=4)
    rend = TopDownRenderer(w.map, cfg["robot"]["fov_deg"], cfg["robot"]["cam_range"][1])
    colors = [TARGET_C if t else OTHER_C for t in w.people.is_target]
    frames = []
    steps = int(seconds / w.dt)
    for k in range(steps):
        w.step(naive_chase(w))
        pp = w.people
        modes = [a.head_mode if a.head_mode != "body" else "" for a in pp.agents]
        txt = (f"M3 world demo | {fam} seed {seed} | t={w.t:4.1f}s | robot: naive GT chase (demo only)\n"
               f"red=target  grey=others  black arrow=body heading  yellow=head yaw  blue wedge=camera FOV")
        frames.append(rend.draw(w.robot.pose, w.robot.camera_pose(), pp.pos, pp.heading, pp.head, colors,
                                labels=modes, text=txt))
    rend.close()
    path = out / f"world_{fam}.gif"
    save_gif(frames, path, fps=10)
    return str(path.relative_to(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "M3"))
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    meta = map_overview(cfg, out)
    gifs = [family_gif(cfg, fam, args.seed, out, args.seconds) for fam in FAMILIES]
    with open(out / "demo_meta.json", "w") as f:
        json.dump({"maps": meta, "gifs": gifs, "seed": args.seed, "seconds": args.seconds}, f, indent=1, default=str)
    print("wrote", out / "maps_overview.png", *gifs, sep="\n  ")


if __name__ == "__main__":
    main()
