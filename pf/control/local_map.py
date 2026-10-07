"""Robot-known obstacle map (Section 8.1): built ONLY from the noisy depth rays, in the odometry frame.

Log-odds-style counts: ray endpoints (< max range) add evidence, cells along the ray lose evidence.
Clearance queries use a distance transform of a small window around the robot.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class LocalMap:
    def __init__(self, dc: dict, origin_xy, max_range: float):
        self.res = dc["local_res"]
        self.size = dc["local_size"]
        self.n = int(round(self.size / self.res))
        self.origin = np.asarray(origin_xy, float) - self.size / 2   # lower-left corner (odom frame)
        self.ev = np.zeros((self.n, self.n), dtype=np.float32)
        self.max_range = max_range
        self.win = int(round(dc["local_window"] / self.res))
        self._clear = None
        self._clear_origin = None

    def _idx(self, xy):
        ij = np.floor((np.asarray(xy) - self.origin) / self.res).astype(np.int64)
        return ij[..., 1], ij[..., 0]

    def occupied(self):
        return self.ev > 0.5

    def update(self, odom, ray_angles, depths):
        """Integrate one depth scan taken from the robot's (estimated) pose."""
        x, y, yaw = odom
        ang = yaw + ray_angles
        c, s = np.cos(ang), np.sin(ang)
        # free space along each ray (sampled at the grid resolution, stopping short of the hit)
        ts = np.arange(0.0, self.max_range, self.res)
        keep = ts[None, :] < (depths[:, None] - 2 * self.res)
        px = (x + c[:, None] * ts[None, :])[keep]
        py = (y + s[:, None] * ts[None, :])[keep]
        if len(px):
            iy, ix = self._idx(np.stack([px, py], axis=1))
            ok = (iy >= 0) & (iy < self.n) & (ix >= 0) & (ix < self.n)
            flat = np.unique(iy[ok] * self.n + ix[ok])
            self.ev.ravel()[flat] -= 0.2
        hit = depths < self.max_range - 1e-6
        if hit.any():
            hx = x + c[hit] * depths[hit]
            hy = y + s[hit] * depths[hit]
            iy, ix = self._idx(np.stack([hx, hy], axis=1))
            ok = (iy >= 0) & (iy < self.n) & (ix >= 0) & (ix < self.n)
            np.add.at(self.ev, (iy[ok], ix[ok]), 1.0)
        np.clip(self.ev, -2.0, 4.0, out=self.ev)
        self._clear = None

    def prepare(self, center_xy):
        """Distance transform of the window around center_xy (call once per control step)."""
        iy, ix = self._idx(center_xy)
        if min(iy, ix) < self.win or max(iy, ix) >= self.n - self.win:
            self._recenter(center_xy)       # robot walked near the edge of the map (large buildings)
            iy, ix = self._idx(center_xy)
        y0, x0 = max(iy - self.win, 0), max(ix - self.win, 0)
        y1, x1 = min(iy + self.win + 1, self.n), min(ix + self.win + 1, self.n)
        occ = self.ev[y0:y1, x0:x1] > 0.5
        if occ.any():
            d = ndimage.distance_transform_edt(~occ) * self.res - 0.5 * self.res
        else:
            d = np.full(occ.shape, 99.0)
        self._clear = np.maximum(d, 0.0)
        self._clear_origin = (y0, x0)

    def _recenter(self, center_xy):
        """Shift the evidence grid so center_xy is in the middle (keeps the overlapping evidence)."""
        new_origin = np.floor((np.asarray(center_xy, float) - self.size / 2) / self.res) * self.res
        sy, sx = (np.round((new_origin - self.origin) / self.res).astype(np.int64))[::-1]
        ev = np.zeros_like(self.ev)
        ys, yd = (sy, 0) if sy >= 0 else (0, -sy)
        xs, xd = (sx, 0) if sx >= 0 else (0, -sx)
        h, w = self.n - abs(sy), self.n - abs(sx)
        if h > 0 and w > 0:
            ev[yd:yd + h, xd:xd + w] = self.ev[ys:ys + h, xs:xs + w]
        self.ev, self.origin = ev, new_origin

    def clearance(self, xy):
        """Distance to the nearest known obstacle (m). Points outside the window get the window edge value."""
        iy, ix = self._idx(xy)
        y0, x0 = self._clear_origin
        h, w = self._clear.shape
        ly = np.clip(iy - y0, 0, h - 1)
        lx = np.clip(ix - x0, 0, w - 1)
        return self._clear[ly, lx]
