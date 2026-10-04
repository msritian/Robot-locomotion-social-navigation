"""Procedural indoor maps (Section 4.2): office, open hall, corridor loop.

Each map is built by carving free rectangles out of a solid block (walls remain), then adding
furniture/pillars. The final occupancy grid is the single source of truth; `obstacles` is its
exact rectangle decomposition (used for 3D extrusion in Part D).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from pf.config import split_of
from pf.world.grid import Grid, decompose_rects, rasterize

FAMILIES = ("office", "hall", "loop")


@dataclass
class Map:
    family: str
    seed: int
    split: str
    grid: Grid
    pois: np.ndarray                  # (M, 2)
    poi_kinds: list[str]
    params: dict = field(default_factory=dict)
    # Free rectangles that were carved (rooms, corridors); useful for scenario scripting.
    regions: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.grid.width

    @property
    def height(self) -> float:
        return self.grid.height

    @property
    def obstacles(self) -> np.ndarray:
        if "_rects" not in self.params:
            self.params["_rects"] = decompose_rects(self.grid.occ, self.grid.res)
        return self.params["_rects"]

    def summary(self) -> dict:
        p = {k: v for k, v in self.params.items() if not k.startswith("_")}
        return {"family": self.family, "seed": self.seed, "split": self.split,
                "size": (round(self.width, 2), round(self.height, 2)), "n_pois": len(self.pois), **p}


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _split_range(spec, split):
    return spec[split] if isinstance(spec, dict) else spec


# ---------------------------------------------------------------------------- office
def _office(rng, cfg, split):
    mc = cfg["maps"]["office"]
    t = cfg["maps"]["wall_thickness"]
    n_rooms = int(rng.integers(mc["n_rooms"][0], mc["n_rooms"][1] + 1))
    n_top = int(np.ceil(n_rooms / 2))
    n_bot = n_rooms - n_top
    wc = _u(rng, _split_range(mc["corridor_w"], split))
    d_top = _u(rng, mc["room_d"])
    d_bot = _u(rng, mc["room_d"]) if n_bot else 0.0
    w_top = [_u(rng, mc["room_w"]) for _ in range(n_top)]
    w_bot = [_u(rng, mc["room_w"]) for _ in range(n_bot)]
    # Corridor extends ~1.5 m past the last room on each end ("corridor ends").
    ext = 1.5
    length = max(sum(w_top) + t * (n_top + 1), sum(w_bot) + t * (n_bot + 1)) + 2 * ext
    y_cor0 = t + (d_bot + t if n_bot else 0.0)
    y_cor1 = y_cor0 + wc
    W = length + 2 * t
    H = y_cor1 + t + d_top + t

    free, occ_extra, pois, kinds, rooms, doors = [], [], [], [], [], []
    corridor = (t, y_cor0, t + length, y_cor1)
    free.append(corridor)

    def add_rooms(widths, y0, y1, door_side):
        x = t + ext + t
        for w in widths:
            room = (x, y0, x + w, y1)
            free.append(room)
            rooms.append(room)
            dw = _u(rng, mc["door_w"])
            dx = float(rng.uniform(x + 0.4, x + w - 0.4 - dw))
            # door: carve through the wall between room and corridor
            if door_side == "below":   # room above corridor, door in its bottom wall
                door = (dx, y_cor1, dx + dw, y0)
                doors.append(((dx + dw / 2, y_cor1 + t / 2), +1))
            else:                      # room below corridor, door in its top wall
                door = (dx, y1, dx + dw, y_cor0)
                doors.append(((dx + dw / 2, y1 + t / 2), -1))
            free.append(door)
            # desks
            n_desks = int(rng.integers(mc["desks_per_room"][0], mc["desks_per_room"][1] + 1))
            for _ in range(n_desks):
                for _try in range(20):
                    horiz = rng.random() < 0.5
                    dl, dd = rng.uniform(1.2, 1.6), rng.uniform(0.6, 0.8)
                    ww, hh = (dl, dd) if horiz else (dd, dl)
                    cx = rng.uniform(room[0] + 0.7 + ww / 2, room[2] - 0.7 - ww / 2) if room[2] - room[0] > ww + 1.4 else None
                    cy = rng.uniform(room[1] + 0.7 + hh / 2, room[3] - 0.7 - hh / 2) if room[3] - room[1] > hh + 1.4 else None
                    if cx is None or cy is None:
                        continue
                    desk = (cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)
                    door_c = doors[-1][0]
                    if np.hypot(np.clip(door_c[0], desk[0], desk[2]) - door_c[0],
                                np.clip(door_c[1], desk[1], desk[3]) - door_c[1]) < 1.5:
                        continue
                    occ_extra.append(desk)
                    # desk front: 0.6 m from the long side facing the room center
                    rc = ((room[0] + room[2]) / 2, (room[1] + room[3]) / 2)
                    if horiz:
                        fy = desk[1] - 0.6 if rc[1] < cy else desk[3] + 0.6
                        pois.append((cx, fy))
                    else:
                        fx = desk[0] - 0.6 if rc[0] < cx else desk[2] + 0.6
                        pois.append((fx, cy))
                    kinds.append("desk_front")
                    break
            pois.append(((room[0] + room[2]) / 2, (room[1] + room[3]) / 2))
            kinds.append("room_center")
            x += w + t

    add_rooms(w_top, y_cor1 + t, y_cor1 + t + d_top, "below")
    if n_bot:
        add_rooms(w_bot, t, t + d_bot, "above")
    for (c, sgn) in doors:
        pois.append((c[0], c[1] + sgn * 0.6))
        kinds.append("doorway_room")
        pois.append((c[0], c[1] - sgn * (t / 2 + min(0.6, wc / 2))))
        kinds.append("doorway_corridor")
    yc = (y_cor0 + y_cor1) / 2
    pois += [(t + 0.6, yc), (t + length - 0.6, yc)]
    kinds += ["corridor_end", "corridor_end"]
    params = {"n_rooms": n_rooms, "corridor_w": round(wc, 3)}
    regions = {"corridor": corridor, "rooms": rooms, "doors": [d[0] for d in doors]}
    return W, H, free, occ_extra, pois, kinds, params, regions


# ---------------------------------------------------------------------------- open hall
def _hall(rng, cfg, split):
    mc = cfg["maps"]["hall"]
    t = cfg["maps"]["wall_thickness"]
    sx, sy = _u(rng, mc["size"]), _u(rng, mc["size"])
    W, H = sx + 2 * t, sy + 2 * t
    free = [(t, t, t + sx, t + sy)]
    occ_extra = []
    spacing = _u(rng, mc["pillar_spacing"])
    ps = _u(rng, mc["pillar_size"])
    xs = np.arange(t + spacing, t + sx - 1.0, spacing)
    ys = np.arange(t + spacing, t + sy - 1.0, spacing)
    for x in xs:
        for y in ys:
            if rng.random() < mc["pillar_keep_p"]:
                occ_extra.append((x - ps / 2, y - ps / 2, x + ps / 2, y + ps / 2))
    n_clusters = int(rng.integers(mc["clusters"][split][0], mc["clusters"][split][1] + 1))
    pois, kinds = [], []
    for _ in range(n_clusters):
        cx, cy = rng.uniform(t + 2.0, t + sx - 2.0), rng.uniform(t + 2.0, t + sy - 2.0)
        kind = rng.choice(["tables", "shelves", "sofa"])
        if kind == "tables":
            for k in range(int(rng.integers(1, 4))):
                w, h = rng.uniform(1.2, 2.0), rng.uniform(0.6, 1.0)
                ox, oy = cx + (k - 1) * (w + 0.9), cy
                occ_extra.append((ox - w / 2, oy - h / 2, ox + w / 2, oy + h / 2))
            pois.append((cx, cy - 1.2)); kinds.append("table_front")
        elif kind == "shelves":
            horiz = rng.random() < 0.5
            for k in range(int(rng.integers(1, 3))):
                L = rng.uniform(1.5, 3.0)
                if horiz:
                    oy = cy + k * 1.6
                    occ_extra.append((cx - L / 2, oy - 0.2, cx + L / 2, oy + 0.2))
                else:
                    ox = cx + k * 1.6
                    occ_extra.append((ox - 0.2, cy - L / 2, ox + 0.2, cy + L / 2))
            pois.append((cx - 1.0, cy - 1.0) if horiz else (cx - 1.0, cy)); kinds.append("shelf_front")
        else:
            w, h = rng.uniform(1.6, 2.2), rng.uniform(0.8, 1.0)
            occ_extra.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            pois.append((cx, cy + h / 2 + 0.7)); kinds.append("sofa_front")
    # Spread-out POIs over the hall (min spacing 3 m) so people cross the space.
    for _ in range(200):
        p = (rng.uniform(t + 1.0, t + sx - 1.0), rng.uniform(t + 1.0, t + sy - 1.0))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 3.0 for q in pois):
            pois.append(p); kinds.append("open_area")
        if len(pois) >= 12:
            break
    params = {"size": (round(sx, 2), round(sy, 2)), "n_clusters": n_clusters, "pillar_spacing": round(spacing, 2)}
    return W, H, free, occ_extra, pois, kinds, params, {"hall": free[0]}


# ---------------------------------------------------------------------------- corridor loop
def _loop(rng, cfg, split):
    mc = cfg["maps"]["loop"]
    t = cfg["maps"]["wall_thickness"]
    ox, oy = _u(rng, mc["outer_x"]), _u(rng, mc["outer_y"])
    w = _u(rng, _split_range(mc["corridor_w"], split))
    fig8 = bool(rng.random() < mc["figure8_p"])
    W, H = ox + 2 * t, oy + 2 * t
    x0, y0, x1, y1 = t, t, t + ox, t + oy
    free = [
        (x0, y0, x1, y0 + w), (x0, y1 - w, x1, y1),   # bottom, top
        (x0, y0, x0 + w, y1), (x1 - w, y0, x1, y1),   # left, right
    ]
    xm = None
    if fig8:
        xm = float(rng.uniform(x0 + 0.4 * ox, x0 + 0.6 * ox))
        free.append((xm - w / 2, y0, xm + w / 2, y1))
    pois, kinds = [], []
    h = w / 2
    corners = [(x0 + h, y0 + h), (x1 - h, y0 + h), (x1 - h, y1 - h), (x0 + h, y1 - h)]
    pois += corners; kinds += ["corner"] * 4
    pois += [((x0 + x1) / 2, y0 + h), ((x0 + x1) / 2, y1 - h), (x0 + h, (y0 + y1) / 2), (x1 - h, (y0 + y1) / 2)]
    kinds += ["corridor_mid"] * 4
    if fig8:
        pois += [(xm, y0 + h), (xm, y1 - h), (xm, (y0 + y1) / 2)]
        kinds += ["junction", "junction", "corridor_mid"]
        # side midpoints would coincide with junctions; shift the top/bottom mids into each lobe
        pois[4] = ((x0 + xm) / 2, y0 + h)
        pois[5] = ((xm + x1) / 2, y1 - h)
    params = {"outer": (round(ox, 2), round(oy, 2)), "corridor_w": round(w, 3), "figure8": fig8}
    return W, H, free, [], pois, kinds, params, {"corridors": free}


_GEN = {"office": _office, "hall": _hall, "loop": _loop}


def generate_map(family: str, seed: int, cfg: dict) -> Map:
    """Deterministic map for (family, seed). Seeds >= 1000 use the held-out (test) parameter ranges."""
    if family not in _GEN:
        raise ValueError(f"unknown map family {family!r}")
    split = split_of(seed, cfg)
    res = cfg["sim"]["grid_res"]
    infl = cfg["people"]["plan_inflation"]
    fam_id = FAMILIES.index(family)
    for attempt in range(50):
        rng = np.random.default_rng([seed, fam_id, attempt])
        W, H, free, occ_extra, pois, kinds, params, regions = _GEN[family](rng, cfg, split)
        occ = rasterize(W, H, res, rects_free=free, rects_occ=occ_extra)
        grid = Grid(occ, res)
        # Keep only POIs in the largest connected free component of the inflated grid.
        walk = ~grid.inflated(infl)
        lab, n = ndimage.label(walk)
        if n == 0:
            continue
        sizes = ndimage.sum(walk, lab, index=np.arange(1, n + 1))
        main = int(np.argmax(sizes)) + 1
        keep_p, keep_k = [], []
        for p, k in zip(pois, kinds):
            q = _snap(grid, lab, main, np.asarray(p, dtype=float))
            if q is not None and all(np.hypot(*(q - r)) > 0.5 for r in keep_p):
                keep_p.append(q); keep_k.append(k)
        if len(keep_p) < 4:
            continue
        # Reject maps where a big chunk of free space is unreachable (e.g. furniture sealing a room).
        if sizes[main - 1] < 0.85 * walk.sum():
            continue
        params["attempt"] = attempt
        return Map(family, seed, split, grid, np.array(keep_p), keep_k, params, regions)
    raise RuntimeError(f"could not generate a valid {family} map for seed {seed}")


def _snap(grid: Grid, lab, main, p, max_dist=0.6):
    """Move p to the nearest cell of the main walkable component within max_dist."""
    iy, ix = grid.to_index(p)
    r = int(np.ceil(max_dist / grid.res))
    y0, y1 = max(iy - r, 0), min(iy + r + 1, grid.ny)
    x0, x1 = max(ix - r, 0), min(ix + r + 1, grid.nx)
    win = lab[y0:y1, x0:x1] == main
    if not win.any():
        return None
    yy, xx = np.nonzero(win)
    cy = (yy + y0 + 0.5) * grid.res
    cx = (xx + x0 + 0.5) * grid.res
    d = np.hypot(cx - p[0], cy - p[1])
    j = int(np.argmin(d))
    if d[j] > max_dist:
        return None
    return np.array([cx[j], cy[j]]) if d[j] > 1e-9 else p
