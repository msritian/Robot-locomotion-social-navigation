"""Top-down GIFs of closed-loop episodes: truth + robot + brain state (goal, forecast, selected track)."""
from __future__ import annotations

import numpy as np

from pf.eval.episode import run_episode
from pf.perception.demo import odom_to_true
from pf.world.render import OTHER_C, TARGET_C, TopDownRenderer, save_gif


def episode_gif(cfg, world, brain, seed, path, title="", steps=None, every=1, noise_scale=None):
    rend = TopDownRenderer(world.map, cfg["robot"]["fov_deg"], cfg["robot"]["cam_range"][1])
    colors = [TARGET_C if t else OTHER_C for t in world.people.is_target]
    colors = ["#ff9896" if (not t and getattr(world, "lookalike_ids", None) and i in world.lookalike_ids) else c
              for i, (t, c) in enumerate(zip(world.people.is_target, colors))]
    frames = []
    k = [0]

    def render(w, b, obs):
        k[0] += 1
        if (k[0] - 1) % every:
            return
        odom, pose = w.robot.odom, w.robot.pose
        ovs = []
        if b.goal is not None:
            ovs.append(("points", odom_to_true(np.asarray(b.goal), odom, pose)[None], dict(marker="*", c="#2ca02c", s=120)))
        if b.forecast is not None and b.state == "follow":
            f = b.forecast
            for m in np.argsort(-f.probs)[:3]:
                if f.probs[m] < 0.05:
                    continue
                pts = np.array([odom_to_true(p, odom, pose) for p in f.modes[m]])
                ovs.append(("line", pts, dict(color="#9467bd", lw=1 + 2 * f.probs[m], alpha=0.8)))
        if b.selected is not None:
            p = odom_to_true(np.asarray(b.selected["pos_world_est"]), odom, pose)
            ovs.append(("points", p[None], dict(marker="o", facecolors="none", edgecolors="#1f77b4", s=260, linewidths=2)))
        sel = b.selected["_gt_id"] if b.selected is not None else None
        ok = "" if sel is None else ("  selected=TARGET" if sel == w.target_index else "  selected=WRONG PERSON")
        d = np.hypot(*(w.people.pos[w.target_index] - pose[:2]))
        txt = f"{title}\nt={w.t:4.1f}s  state={b.state}  dist={d:.2f}m{ok}"
        frames.append(rend.draw(pose, w.robot.camera_pose(), w.people.pos, w.people.heading, w.people.head, colors,
                                text=txt, overlays=ovs))

    log = run_episode(cfg, world, brain, seed, noise_scale=noise_scale, steps=steps, render=render)
    rend.close()
    save_gif(frames, path, fps=10 / every)
    return log
