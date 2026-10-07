"""InteriorGS / SAGE-3D scenes -> Stage C maps (PROJECT_SPEC 19.2, "Use 1" and step 3 of "Use 2").

Source: SAGE-3D collision meshes (spatialverse/SAGE-3D_Collision_Mesh, CC-BY-NC-4.0), static colliders incl.
the floor plan (walls, floor), doors, windows and furniture. The occupancy grid is the horizontal slice of all
collision triangles between 0.1 and 1.5 m height (Section 19.2 allows this when the gated InteriorGS walkable
maps are not used). Doors are left out so doorways stay passable. Space outside the building (connected to the
grid border) is marked occupied.

Frame: SAGE-3D's scene template rotates the collision payload by 180 deg about z to align it with the splat;
we apply the same rotation, then shift so the map's lower-left corner is the origin. `Map.params["frame"]`
stores (shift_x, shift_y) so Isaac positions = map positions - shift (both after the 180 deg rotation).

POIs: the hashed object names carry no semantics without the gated InteriorGS labels, so POIs are spread-out
walkable points (>= 3 m apart) plus points in front of large objects (desk/counter-like boxes 0.5-1.2 m tall).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage

from pf.world.grid import Grid
from pf.world.maps import Map, _snap

Z_LO, Z_HI = 0.1, 1.5


def _mesh_triangles(path):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(path))
    tris, names = [], []
    xc = UsdGeom.XformCache()
    for prim in st.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        name = prim.GetName()
        parent = prim.GetParent().GetName()
        if name.lower().startswith("door") or parent.lower().startswith("door"):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        if pts.size == 0:
            continue
        M = np.asarray(xc.GetLocalToWorldTransform(prim), dtype=np.float64)   # row-vector convention
        pts = pts @ M[:3, :3] + M[3, :3]
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        # fan-triangulate polygons
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri = []
        for s, c in zip(starts, counts):
            for k in range(1, c - 1):
                tri.append((idx[s], idx[s + k], idx[s + k + 1]))
        if tri:
            t = pts[np.asarray(tri)]
            tris.append(t)
            names.extend([name] * len(t))
    return np.concatenate(tris), np.asarray(names)


def _rasterize(tris, res, x0, y0, nx, ny):
    occ = np.zeros((ny, nx), dtype=bool)
    # densely sample each triangle (barycentric grid with spacing <= res/2), project to xy
    e = np.maximum.reduce([np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1), np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1),
                           np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)])
    k = np.clip(np.ceil(e / (0.5 * res)).astype(int), 1, 400)
    for kk in np.unique(k):
        sel = tris[k == kk]
        a = np.linspace(0, 1, kk + 1)
        u, v = np.meshgrid(a, a)
        m = (u + v) <= 1.0
        u, v = u[m], v[m]
        p = sel[:, None, 0] + u[None, :, None] * (sel[:, None, 1] - sel[:, None, 0]) + \
            v[None, :, None] * (sel[:, None, 2] - sel[:, None, 0])
        p = p.reshape(-1, 3)
        p = p[(p[:, 2] >= Z_LO) & (p[:, 2] <= Z_HI)]
        ix = np.floor((p[:, 0] - x0) / res).astype(int)
        iy = np.floor((p[:, 1] - y0) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ[iy[ok], ix[ok]] = True
    return occ


def _rasterize_xy(tris, res, x0, y0, nx, ny):
    """Rasterize triangles' xy footprints regardless of height."""
    t = tris.copy()
    t[..., 2] = 0.5 * (Z_LO + Z_HI)
    return _rasterize(t, res, x0, y0, nx, ny)


def scene_to_map(collision_usd, scene_id, cfg, split="test", res=None, margin=0.5):
    res = res or cfg["sim"]["grid_res"]
    tris, names = _mesh_triangles(collision_usd)
    tris = tris.copy()
    tris[..., :2] *= -1.0                       # 180 deg about z (SAGE-3D template alignment)
    zmin, zmax = tris[..., 2].min(axis=1), tris[..., 2].max(axis=1)
    band = (zmax >= Z_LO) & (zmin <= Z_HI)
    lo = tris[..., :2].reshape(-1, 2).min(0) - margin
    hi = tris[..., :2].reshape(-1, 2).max(0) + margin
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    occ = _rasterize(tris[band], res, lo[0], lo[1], nx, ny)
    # inside of the building = cells covered by floor triangles (horizontal, near z = 0); everything else is
    # occupied. (Flood-filling from the border leaks through entrances once doors are removed.)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    horiz = np.abs(nrm[:, 2]) > 0.9 * np.linalg.norm(nrm, axis=1)
    floor = horiz & (zmax < Z_LO) & (zmin > -0.5)
    fl = _rasterize_xy(tris[floor], res, lo[0], lo[1], nx, ny)
    fl = ndimage.binary_closing(fl, iterations=2)
    occ |= ~fl
    grid = Grid(occ, res)
    # largest walkable component (0.3 m inflation), POIs spread >= 3 m apart + object fronts
    walk = ~grid.inflated(cfg["people"]["plan_inflation"])
    lab, n = ndimage.label(walk)
    if n == 0:
        raise RuntimeError(f"scene {scene_id}: no walkable space")
    sizes = ndimage.sum(walk, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    rng = np.random.default_rng(int(scene_id))
    iy, ix = np.nonzero(lab == main)
    cand = np.stack([(ix + 0.5) * res, (iy + 0.5) * res], 1)
    pois = []
    for p in cand[rng.permutation(len(cand))]:
        if all(np.hypot(*(p - q)) >= 3.0 for q in pois):
            pois.append(p)
        if len(pois) >= 20:
            break
    kinds = ["open_area"] * len(pois)
    keep_p, keep_k = [], []
    for p, k in zip(pois, kinds):
        q = _snap(grid, lab, main, np.asarray(p))
        if q is not None:
            keep_p.append(q)
            keep_k.append(k)
    walk_area = float(sizes[main - 1] * res * res)
    params = {"source": "SAGE-3D collision mesh slice", "scene_id": int(scene_id), "walkable_m2": round(walk_area, 1),
              "frame": (float(lo[0]), float(lo[1])), "rot_deg": 180.0, "z_band": (Z_LO, Z_HI)}
    return Map("interiorgs", int(scene_id), split, grid, np.array(keep_p), keep_k, params, {})


if __name__ == "__main__":
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pf.config import load_config
    cfg = load_config()
    files = sorted(Path(sys.argv[1]).glob("*.usd"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        sid = f.stem.split("_")[0]
        try:
            m = scene_to_map(f, sid, cfg)
        except Exception as e:  # report, keep going
            print(sid, "FAILED", e)
            continue
        rows.append((sid, m.width, m.height, m.params["walkable_m2"], len(m.pois)))
        fig, ax = plt.subplots(figsize=(6, 6 * m.height / max(m.width, 1e-6)))
        ax.imshow(m.grid.occ, origin="lower", extent=(0, m.width, 0, m.height), cmap="Greys", vmin=0, vmax=1.4)
        ax.scatter(m.pois[:, 0], m.pois[:, 1], s=8, c="#2ca02c")
        ax.set_title(f"{sid}: {m.width:.1f} x {m.height:.1f} m, walkable {m.params['walkable_m2']} m2", fontsize=8)
        fig.savefig(out / f"{sid}.png", dpi=90)
        plt.close(fig)
        print(sid, f"{m.width:.1f} x {m.height:.1f} m  walkable {m.params['walkable_m2']} m2  pois {len(m.pois)}", flush=True)


def map_from_npz(path, name, cfg, n_pois=20):
    """Stage C Map from an occupancy grid sliced from an Isaac environment (isaac_follow/env_to_map.py).
    Map coordinates = world coordinates - origin (the environment is placed at -origin in Isaac)."""
    d = np.load(path, allow_pickle=True)
    occ, res = d["occ"].astype(bool), float(d["res"])
    grid = Grid(occ, res)
    walk = ~grid.inflated(cfg["people"]["plan_inflation"])
    lab, n = ndimage.label(walk)
    sizes = ndimage.sum(walk, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    rng = np.random.default_rng(zlib.crc32(name.encode()))   # stable across runs (hash() is salted)
    iy, ix = np.nonzero(lab == main)
    cand = np.stack([(ix + 0.5) * res, (iy + 0.5) * res], 1)
    pois = []
    for p in cand[rng.permutation(len(cand))]:
        if all(np.hypot(*(p - q)) >= 3.0 for q in pois):
            pois.append(p)
        if len(pois) >= n_pois:
            break
    params = {"source": f"Isaac environment {name}", "walkable_m2": round(float(sizes[main - 1] * res * res), 1),
              "frame": tuple(float(v) for v in d["origin"]), "floor_z": float(d["floor_z"]), "url": str(d["url"])}
    return Map("isaac_env", 0, "test", grid, np.array(pois), ["open_area"] * len(pois), params, {})
