"""Top-down matplotlib renderer for Stage C episodes -> GIF frames.

Draws obstacles, POIs, people (body circle, body-heading arrow, head-yaw tick), the target (red),
the robot (blue) with its camera FOV wedge, and optional overlays (tracks, predictions, text).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

TARGET_C = "#d62728"
OTHER_C = "#7f7f7f"
LOOKALIKE_C = "#ff9896"
ROBOT_C = "#1f77b4"


class TopDownRenderer:
    def __init__(self, world_map, fov_deg=90.0, cam_range=8.0, px_per_m=28, title=""):
        self.map = world_map
        self.fov = fov_deg
        self.cam_range = cam_range
        W, H = world_map.width, world_map.height
        dpi = 100
        self.fig = plt.figure(figsize=(max(W * px_per_m / dpi, 6.5), max(H * px_per_m / dpi, 3) + 0.4), dpi=dpi)
        self.ax = self.fig.add_axes([0.01, 0.01, 0.98, 0.9])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.imshow(world_map.grid.occ, origin="lower", extent=(0, W, 0, H), cmap="Greys", vmin=0, vmax=1.4,
                       interpolation="nearest")
        self.ax.scatter(world_map.pois[:, 0], world_map.pois[:, 1], s=6, c="#2ca02c", alpha=0.5, zorder=1)
        self.title = self.fig.text(0.02, 0.93, title, fontsize=7, family="monospace")
        self.dyn = []

    def _clear(self):
        for a in self.dyn:
            a.remove()
        self.dyn = []

    def draw(self, robot_pose, cam_pose, people_pos, people_heading, people_head, colors, labels=None,
             text="", overlays=None):
        self._clear()
        ax = self.ax
        r = 0.25
        for i, (p, hdg, hd, c) in enumerate(zip(people_pos, people_heading, people_head, colors)):
            self.dyn.append(ax.add_patch(Circle(p, r, color=c, zorder=4, alpha=0.9)))
            self.dyn.append(ax.arrow(p[0], p[1], 0.45 * np.cos(hdg), 0.45 * np.sin(hdg), width=0.03,
                                     head_width=0.12, color="k", zorder=5, length_includes_head=True))
            self.dyn += ax.plot([p[0], p[0] + 0.32 * np.cos(hd)], [p[1], p[1] + 0.32 * np.sin(hd)],
                                color="#ffdd00", lw=2.2, zorder=6)
            if labels is not None and labels[i]:
                self.dyn.append(ax.text(p[0] + 0.3, p[1] + 0.3, labels[i], fontsize=6, zorder=7))
        x, y, yaw = cam_pose
        self.dyn.append(ax.add_patch(Wedge((x, y), self.cam_range, np.rad2deg(yaw) - self.fov / 2,
                                           np.rad2deg(yaw) + self.fov / 2, color=ROBOT_C, alpha=0.07, zorder=2)))
        rx, ry, ryaw = robot_pose
        self.dyn.append(ax.add_patch(Circle((rx, ry), 0.2, color=ROBOT_C, zorder=4)))
        self.dyn += ax.plot([rx, rx + 0.4 * np.cos(ryaw)], [ry, ry + 0.4 * np.sin(ryaw)], color="w", lw=2, zorder=5)
        for ov in overlays or []:
            kind = ov[0]
            if kind == "line":
                _, pts, kw = ov
                pts = np.asarray(pts)
                self.dyn += ax.plot(pts[:, 0], pts[:, 1], zorder=3, **kw)
            elif kind == "points":
                _, pts, kw = ov
                pts = np.asarray(pts).reshape(-1, 2)
                self.dyn.append(ax.scatter(pts[:, 0], pts[:, 1], zorder=6, **kw))
        self.title.set_text(text)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return buf.copy()

    def close(self):
        plt.close(self.fig)


def save_gif(frames, path, fps=10):
    imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)
