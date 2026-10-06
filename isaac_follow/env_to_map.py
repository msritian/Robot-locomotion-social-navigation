"""Slice an Isaac Sim environment USD (office / hospital / warehouse, loaded from the Isaac asset server) into a
Stage C occupancy map: all mesh triangles with z in [0.1, 1.5] m are obstacles; the interior is the footprint of
floor-like (horizontal, z < 0.1 m) triangles. Saves <name>_map.npz (occ, res, origin) + a PNG, and an overhead render.

    python isaac_follow/env_to_map.py --name office --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

ENVS = {
    "office": "Isaac/Environments/Office/office.usd",
    "hospital": "Isaac/Environments/Hospital/hospital.usd",
    "warehouse": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
}
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, choices=list(ENVS))
ap.add_argument("--out", default="env_maps")
ap.add_argument("--res", type=float, default=0.05)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy import ndimage  # noqa: E402

os.makedirs(args.out, exist_ok=True)
url = ISAAC_NUCLEUS_DIR.rsplit("/Isaac", 1)[0] + "/" + ENVS[args.name]
print(f"[map] opening {url}", flush=True)
ctx = omni.usd.get_context()
ctx.open_stage(url)
stage0 = ctx.get_stage()
stage0.Load()                         # load all payloads
for _ in range(600):
    app.update()
    if ctx.get_stage_loading_status()[2] == 0:
        break
stage = ctx.get_stage()
mpu = UsdGeom.GetStageMetersPerUnit(stage)
up = UsdGeom.GetStageUpAxis(stage)
print(f"[map] metersPerUnit {mpu} upAxis {up}", flush=True)
xc = UsdGeom.XformCache()
tris = []
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):   # include instanced props
    if prim.GetTypeName() != "Mesh" or not UsdGeom.Imageable(prim).ComputeVisibility() == "inherited":
        continue
    m = UsdGeom.Mesh(prim)
    pts = m.GetPointsAttr().Get()
    cnt = m.GetFaceVertexCountsAttr().Get()
    idx = m.GetFaceVertexIndicesAttr().Get()
    if not pts or not cnt:
        continue
    pts = np.asarray(pts, dtype=np.float64)
    M = np.asarray(xc.GetLocalToWorldTransform(prim), dtype=np.float64)
    pts = (pts @ M[:3, :3] + M[3, :3]) * mpu
    if up == "Y":
        pts = pts[:, [0, 2, 1]] * np.array([1, -1, 1])
    if np.ptp(pts[:, 0]) > 120 or np.ptp(pts[:, 1]) > 120:   # skip ground planes / sky domes / outdoor terrain
        print(f"[map] skipping huge mesh {prim.GetPath()} ({np.ptp(pts[:, 0]):.0f} x {np.ptp(pts[:, 1]):.0f} m)", flush=True)
        continue
    cnt, idx = np.asarray(cnt), np.asarray(idx)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    t = [(idx[s], idx[s + k], idx[s + k + 1]) for s, c in zip(starts, cnt) for k in range(1, c - 1)]
    if t:
        tris.append(pts[np.asarray(t)])
tris = np.concatenate(tris)
print(f"[map] {len(tris)} triangles", flush=True)
zmin, zmax = tris[..., 2].min(1), tris[..., 2].max(1)
# floor height = the z with the most horizontal surface area (robust to basements / outdoor terrain below)
_nn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
_area = 0.5 * np.linalg.norm(_nn, axis=1)
_h = (np.abs(_nn[:, 2]) > 0.95 * np.linalg.norm(_nn, axis=1) + 1e-12) & (zmax - zmin < 0.02)
_zb = np.round(zmin[_h] / 0.05).astype(int)
_u, _inv = np.unique(_zb, return_inverse=True)
floor_z = float(_u[np.argmax(np.bincount(_inv, weights=_area[_h]))] * 0.05) if _h.any() else float(np.percentile(zmin, 5))
print(f"[map] floor_z by horizontal area: {floor_z:.3f}", flush=True)
_w = np.bincount(_inv, weights=_area[_h]) if _h.any() else np.zeros(0)
_up = np.bincount(_inv, weights=_area[_h] * (_nn[_h, 2] > 0)) if _h.any() else np.zeros(0)
for _k in np.argsort(-_w)[:10]:
    _m = _h.copy(); _m[_h] = _inv == _k
    _xy = tris[_m][..., :2].reshape(-1, 2)
    print(f"[map]   level z={_u[_k] * 0.05:7.2f} area={_w[_k]:9.1f} up={_up[_k] / _w[_k]:.2f} n={_m.sum():6d} "
          f"maxtri={_area[_m].max():8.1f} bbox={_xy.min(0).round(1).tolist()}..{_xy.max(0).round(1).tolist()}", flush=True)
tris[..., 2] -= floor_z
zmin, zmax = zmin - floor_z, zmax - floor_z
res = args.res
# map extent from the building's floor (outdoor props/terrain far away would blow up the grid)
_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
_fl = (np.abs(_n[:, 2]) > 0.9 * np.linalg.norm(_n, axis=1)) & (zmax < 0.1) & (zmin > -0.3)
_c = tris[_fl].mean(axis=1)[:, :2] if _fl.sum() > 10 else tris.mean(axis=1)[:, :2]
lo = np.percentile(_c, 0.5, axis=0) - 2.0
hi = np.percentile(_c, 99.5, axis=0) + 2.0
_inside = np.all((tris[..., :2] >= lo) & (tris[..., :2] <= hi), axis=(1, 2))
tris, zmin, zmax = tris[_inside], zmin[_inside], zmax[_inside]
print(f"[map] extent from floor: {lo.round(1).tolist()} .. {hi.round(1).tolist()}, {_inside.sum()} triangles kept", flush=True)
nx, ny = np.ceil((hi - lo) / res).astype(int)


def raster(sel, use_band):
    occ = np.zeros((ny, nx), bool)
    T = tris[sel]
    e = np.maximum.reduce([np.linalg.norm(T[:, i] - T[:, (i + 1) % 3], axis=1) for i in range(3)])
    k = np.clip(np.ceil(e / (0.5 * res)).astype(int), 1, 400)
    for kk in np.unique(k):
        S = T[k == kk]
        a = np.linspace(0, 1, kk + 1)
        u, v = np.meshgrid(a, a)
        mk = (u + v) <= 1
        u, v = u[mk], v[mk]
        p = (S[:, None, 0] + u[None, :, None] * (S[:, None, 1] - S[:, None, 0]) +
             v[None, :, None] * (S[:, None, 2] - S[:, None, 0])).reshape(-1, 3)
        if use_band:
            p = p[(p[:, 2] >= 0.1) & (p[:, 2] <= 1.5)]
        ix = ((p[:, 0] - lo[0]) / res).astype(int)
        iy = ((p[:, 1] - lo[1]) / res).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ[iy[ok], ix[ok]] = True
    return occ


nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
horiz = np.abs(nrm[:, 2]) > 0.9 * np.linalg.norm(nrm, axis=1)
floor = raster(horiz & (zmax < 0.1) & (zmin > -0.3), False)
floor = ndimage.binary_closing(floor, iterations=2)
occ = raster((zmax >= 0.1) & (zmin <= 1.5), True) | ~floor
np.savez_compressed(os.path.join(args.out, f"{args.name}_map.npz"), occ=occ, res=res, origin=lo, floor_z=floor_z,
                    url=url)
import imageio.v2 as imageio  # noqa: E402
imageio.imwrite(os.path.join(args.out, f"{args.name}_map.png"), (np.flipud(~occ) * 255).astype(np.uint8))
print(f"[map] saved {args.name}: {nx}x{ny} cells ({nx * res:.1f} x {ny * res:.1f} m), free {(~occ).mean():.2f}, "
      f"origin {lo.tolist()}, floor_z {floor_z:.3f}", flush=True)
import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
