"""Occupancy-grid utilities: rasterization, distance field, ray casting, line of sight.

Convention: world origin (0, 0) is the map's lower-left corner. Cell (iy, ix) covers
x in [ix*res, (ix+1)*res), y in [iy*res, (iy+1)*res). Arrays are indexed [iy, ix].
Everything outside the grid counts as occupied.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class Grid:
    def __init__(self, occ: np.ndarray, res: float):
        self.occ = occ.astype(bool)
        self.res = float(res)
        self.ny, self.nx = occ.shape
        self.width = self.nx * res
        self.height = self.ny * res
        # Distance (m) from each free cell center to the nearest occupied cell boundary (approx.).
        d = ndimage.distance_transform_edt(~self.occ) * res - 0.5 * res
        self.dist = np.where(self.occ, 0.0, np.maximum(d, 0.0)).astype(np.float32)
        gy, gx = np.gradient(self.dist, res)
        self.grad = np.stack([gx, gy], axis=-1).astype(np.float32)  # points away from obstacles

    # ------------------------------------------------------------------ lookups
    def to_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64)
        ix = np.floor(xy[..., 0] / self.res).astype(np.int64)
        iy = np.floor(xy[..., 1] / self.res).astype(np.int64)
        return iy, ix

    def inside(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def is_occupied(self, xy: np.ndarray) -> np.ndarray:
        iy, ix = self.to_index(xy)
        ok = self.inside(iy, ix)
        out = np.ones(iy.shape, dtype=bool)
        out[ok] = self.occ[iy[ok], ix[ok]]
        return out

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        """Distance (m) to the nearest obstacle; 0 inside obstacles or outside the map."""
        iy, ix = self.to_index(xy)
        ok = self.inside(iy, ix)
        out = np.zeros(iy.shape, dtype=np.float64)
        out[ok] = self.dist[iy[ok], ix[ok]]
        return out

    def clearance_grad(self, xy: np.ndarray) -> np.ndarray:
        iy, ix = self.to_index(xy)
        ok = self.inside(iy, ix)
        out = np.zeros(iy.shape + (2,), dtype=np.float64)
        out[ok] = self.grad[iy[ok], ix[ok]]
        return out

    # ------------------------------------------------------------------ rays
    def raycast(self, origin: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Distance from `origin` (2,) along each world-frame angle to the first occupied cell.

        Fixed-step sampling at half the grid resolution (accuracy ~0.025 m), vectorized over rays.
        """
        angles = np.asarray(angles, dtype=np.float64)
        step = 0.5 * self.res
        t = np.arange(1, int(np.ceil(max_range / step)) + 1) * step
        px = origin[0] + np.cos(angles)[:, None] * t[None, :]
        py = origin[1] + np.sin(angles)[:, None] * t[None, :]
        ix = np.floor(px / self.res).astype(np.int64)
        iy = np.floor(py / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        hit = np.ones(ix.shape, dtype=bool)
        hit[ok] = self.occ[iy[ok], ix[ok]]
        first = np.argmax(hit, axis=1)
        any_hit = hit[np.arange(len(angles)), first]
        return np.where(any_hit, t[first] - 0.5 * step, max_range).clip(0.0, max_range)

    def segment_free(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """True where the segment(s) a->b keep at least `margin` clearance. a, b: (..., 2)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        length = np.linalg.norm(b - a, axis=-1)
        n = int(np.ceil(np.max(length) / (0.5 * self.res))) + 1 if length.size else 1
        s = np.linspace(0.0, 1.0, max(n, 2))
        pts = a[..., None, :] + (b - a)[..., None, :] * s[:, None]
        c = self.clearance(pts)
        return np.all(c > margin, axis=-1)

    def inflated_at(self, xy, radius: float) -> bool:
        return bool(self.clearance(np.asarray(xy, dtype=np.float64)) < radius)

    def inflated(self, radius: float) -> np.ndarray:
        """Occupancy inflated by `radius` meters."""
        return self.occ | (self.dist < radius)


def rasterize(width: float, height: float, res: float, rects_free=(), rects_occ=(),
              solid_background: bool = True) -> np.ndarray:
    """Build an occupancy grid. Start solid (or empty), carve free rects, then add occupied rects."""
    nx, ny = int(np.ceil(width / res)), int(np.ceil(height / res))
    occ = np.full((ny, nx), solid_background, dtype=bool)

    def span(lo, hi, n):
        i0 = int(np.clip(np.round(lo / res), 0, n))
        i1 = int(np.clip(np.round(hi / res), 0, n))
        return i0, i1

    for x0, y0, x1, y1 in rects_free:
        a, b = span(x0, x1, nx)
        c, d = span(y0, y1, ny)
        occ[c:d, a:b] = False
    for x0, y0, x1, y1 in rects_occ:
        a, b = span(x0, x1, nx)
        c, d = span(y0, y1, ny)
        occ[c:d, a:b] = True
    return occ


def decompose_rects(occ: np.ndarray, res: float) -> np.ndarray:
    """Exact decomposition of an occupancy grid into axis-aligned rectangles (x0, y0, x1, y1).

    Greedy: horizontal runs per row, merged with identical runs in following rows.
    Used to export the map as box obstacles (e.g. for the 3D Isaac scene).
    """
    ny, nx = occ.shape
    open_runs: dict[tuple[int, int], int] = {}  # (x0, x1) -> start row
    rects = []
    for iy in range(ny + 1):
        runs = set()
        if iy < ny:
            row = np.concatenate([[False], occ[iy], [False]])
            diff = np.diff(row.astype(np.int8))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            runs = set(zip(starts.tolist(), ends.tolist()))
        for run in list(open_runs):
            if run not in runs:
                y0 = open_runs.pop(run)
                rects.append((run[0] * res, y0 * res, run[1] * res, iy * res))
        for run in runs:
            if run not in open_runs:
                open_runs[run] = iy
    return np.array(rects, dtype=np.float64).reshape(-1, 4)
