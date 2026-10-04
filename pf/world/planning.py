"""A* path planning for simulated people (Section 4.4), with string-pulling smoothing.

A* runs on a 0.1 m grid derived conservatively from the 0.05 m occupancy grid inflated by 0.3 m.
Paths between POIs are cached per map.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from pf.world.grid import Grid

_SQRT2 = math.sqrt(2.0)
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


class PathPlanner:
    def __init__(self, grid: Grid, inflation: float = 0.3, plan_res: float = 0.1):
        self.grid = grid
        self.inflation = inflation
        f = max(1, int(round(plan_res / grid.res)))
        self.f = f
        self.res = grid.res * f
        infl = grid.inflated(inflation)
        ny, nx = infl.shape[0] // f, infl.shape[1] // f
        # Conservative downsample: a coarse cell is blocked if any fine cell is.
        self.blocked = infl[: ny * f, : nx * f].reshape(ny, f, nx, f).any(axis=(1, 3))
        self.ny, self.nx = self.blocked.shape
        self._cache: dict = {}

    def _cell(self, p):
        return (int(np.clip(p[1] // self.res, 0, self.ny - 1)), int(np.clip(p[0] // self.res, 0, self.nx - 1)))

    def _center(self, c):
        return np.array([(c[1] + 0.5) * self.res, (c[0] + 0.5) * self.res])

    def _nearest_free(self, c, max_r=8):
        if not self.blocked[c]:
            return c
        for r in range(1, max_r + 1):
            best = None
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dy), abs(dx)) != r:
                        continue
                    y, x = c[0] + dy, c[1] + dx
                    if 0 <= y < self.ny and 0 <= x < self.nx and not self.blocked[y, x]:
                        d = dy * dy + dx * dx
                        if best is None or d < best[0]:
                            best = (d, (y, x))
            if best:
                return best[1]
        return None

    def astar(self, start, goal):
        s = self._nearest_free(self._cell(start))
        g = self._nearest_free(self._cell(goal))
        if s is None or g is None:
            return None
        blocked = self.blocked
        ny, nx = self.ny, self.nx
        gy, gx = g

        def h(y, x):
            dy, dx = abs(y - gy), abs(x - gx)
            return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)

        open_heap = [(h(*s), 0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while open_heap:
            _, c, cur = heapq.heappop(open_heap)
            if cur == g:
                break
            if c > cost[cur]:
                continue
            y, x = cur
            for dy, dx, w in _NBRS:
                yy, xx = y + dy, x + dx
                if not (0 <= yy < ny and 0 <= xx < nx) or blocked[yy, xx]:
                    continue
                if dy and dx and (blocked[y, xx] or blocked[yy, x]):
                    continue  # no corner cutting
                nc = c + w
                nb = (yy, xx)
                if nc < cost.get(nb, math.inf):
                    cost[nb] = nc
                    came[nb] = cur
                    heapq.heappush(open_heap, (nc + h(yy, xx), nc, nb))
        if g not in came:
            return None
        cells = []
        cur = g
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        cells.reverse()
        pts = np.array([self._center(c) for c in cells])
        pts[0] = start if not self.grid.inflated_at(start, self.inflation) else pts[0]
        pts[-1] = goal if not self.grid.inflated_at(goal, self.inflation) else pts[-1]
        return self.smooth(pts)

    def smooth(self, pts: np.ndarray) -> np.ndarray:
        """String pulling: keep a vertex only when the straight shortcut would violate clearance."""
        if len(pts) <= 2:
            return pts
        margin = self.inflation - 0.5 * self.grid.res
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            # extend forward while the shortcut from i stays clear
            j = i + 1
            while j + 1 < n and self.grid.segment_free(pts[i], pts[j + 1], margin):
                j += 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def path(self, start, goal, key=None):
        """Planned path start->goal; cached when `key` (e.g. (poi_i, poi_j)) is given."""
        if key is not None and key in self._cache:
            return self._cache[key]
        p = self.astar(np.asarray(start, float), np.asarray(goal, float))
        if key is not None:
            self._cache[key] = p
        return p
