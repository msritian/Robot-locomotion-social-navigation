"""Offline retargeting of Isaac Sim's biped animation clips (Root/Pelvis/R_UpLeg..., 81 joints) onto the
Reallusion-skeleton characters (RL_BoneRoot/Hip/Pelvis/L_Thigh..., 101 joints) used by Isaac/People/Characters.

Method (global rest-relative rotation transfer):
  D_s(t) = G_s(t) * G_s_rest^-1          world-space rotation of each mapped source bone relative to its rest pose
  G_t(t) = A * D_s(t) * A^-1 * G_t_rest  apply it to the target bone's rest world rotation (A aligns the skeletons'
                                          body frames: left-right axis and up axis)
  L_t(t) = G_t(parent)^-1 * G_t(t)        back to local; unmapped target joints keep their rest local transform.
Hip translation follows the source pelvis offset from rest, scaled by the leg-length ratio.

Output: isaac_follow/anim/<character>_<clip>.npz with joints, per-frame local translations / rotations (w,x,y,z) /
scales, ready to be written into a UsdSkel.Animation each frame (pure playback at runtime).

    python -m isaac_follow.retarget <assets_dir_with_usd_files> [--plot]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdSkel

MAP = {
    "Root": "RL_BoneRoot", "Pelvis": "Hip",
    "R_UpLeg": "R_Thigh", "R_LoLeg": "R_Calf", "R_Ankle": "R_Foot", "R_Ball": "R_ToeBase",
    "L_UpLeg": "L_Thigh", "L_LoLeg": "L_Calf", "L_Ankle": "L_Foot", "L_Ball": "L_ToeBase",
    "Spine1": "Waist", "Spine2": "Spine01", "Chest": "Spine02", "Neck1": "NeckTwist01", "Neck2": "NeckTwist02",
    "Head": "Head",
    "L_Clavicle": "L_Clavicle", "L_UpArm": "L_Upperarm", "L_LoArm": "L_Forearm", "L_Wrist": "L_Hand",
    "R_Clavicle": "R_Clavicle", "R_UpArm": "R_Upperarm", "R_LoArm": "R_Forearm", "R_Wrist": "R_Hand",
}


def _skeleton(path):
    st = Usd.Stage.Open(str(path))
    for p in st.Traverse():
        if p.IsA(UsdSkel.Skeleton):
            sk = UsdSkel.Skeleton(p)
            joints = [str(j) for j in sk.GetJointsAttr().Get()]
            rest = [np.array(m, dtype=np.float64) for m in sk.GetRestTransformsAttr().Get()]
            return joints, rest
    raise RuntimeError(f"no skeleton in {path}")


def _parents(joints):
    idx = {j: i for i, j in enumerate(joints)}
    return [idx.get(j.rsplit("/", 1)[0], -1) if "/" in j else -1 for j in joints]


def _world(local, parents):
    """local: list of 4x4 row-vector matrices (USD convention: v' = v * M). World = local * parent_world."""
    w = [None] * len(local)
    for i, m in enumerate(local):
        w[i] = m if parents[i] < 0 else m @ w[parents[i]]
    return w


def _rot(m):
    r = m[:3, :3].copy()
    r /= np.linalg.norm(r, axis=1, keepdims=True)   # remove scale (row-wise)
    return r


def _mat(rot, trans, scale=(1, 1, 1)):
    m = np.eye(4)
    m[:3, :3] = np.diag(scale) @ rot
    m[3, :3] = trans
    return m


def _quat_to_rot(q):
    """(w, x, y, z) -> 3x3 rotation for ROW vectors (USD/Gf convention: v' = v @ M, M = R_column^T)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    rc = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return rc.T


def _rot_to_quat(r):
    """Inverse of _quat_to_rot (row-vector matrix -> (w, x, y, z))."""
    m = np.asarray(r, dtype=np.float64).T
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.array(q)


def _body_frame(world, joints, lr_pair, up_pair):
    """Rows: left-right axis, up axis, their cross (body frame) from rest joint positions."""
    pos = {j.split("/")[-1]: world[i][3, :3] for i, j in enumerate(joints)}
    lr = pos[lr_pair[0]] - pos[lr_pair[1]]
    up = pos[up_pair[0]] - pos[up_pair[1]]
    lr /= np.linalg.norm(lr)
    up -= lr * (up @ lr)
    up /= np.linalg.norm(up)
    return np.stack([lr, up, np.cross(lr, up)])


def load_clip(path):
    st = Usd.Stage.Open(str(path))
    anim = UsdSkel.Animation(st.GetDefaultPrim())
    times = np.arange(st.GetStartTimeCode(), st.GetEndTimeCode() + 1e-6)
    return {"joints": [str(j) for j in anim.GetJointsAttr().Get()], "fps": st.GetTimeCodesPerSecond(),
            "t": [np.array(anim.GetTranslationsAttr().Get(t)) for t in times],
            "r": [np.array([[q.GetReal(), *q.GetImaginary()] for q in anim.GetRotationsAttr().Get(t)]) for t in times],
            "s": [np.array(anim.GetScalesAttr().Get(t)) for t in times]}


def retarget(src_skel, clip, tgt_skel):
    sj, srest = src_skel
    tj, trest = tgt_skel
    sp, tp = _parents(sj), _parents(tj)
    s_world_rest = _world(srest, sp)
    t_world_rest = _world(trest, tp)
    sname = {j.split("/")[-1]: i for i, j in enumerate(sj)}
    tname = {j.split("/")[-1]: i for i, j in enumerate(tj)}
    # align body frames: A maps source body-frame coordinates to target ones
    Fs = _body_frame(s_world_rest, sj, ("L_UpLeg", "R_UpLeg"), ("Head", "Pelvis"))
    Ft = _body_frame(t_world_rest, tj, ("L_Thigh", "R_Thigh"), ("Head", "Hip"))
    A = Fs.T @ Ft                         # row-vector convention: v_t = v_s @ A
    leg_s = np.linalg.norm(s_world_rest[sname["L_UpLeg"]][3, :3] - s_world_rest[sname["L_Ankle"]][3, :3])
    leg_t = np.linalg.norm(t_world_rest[tname["L_Thigh"]][3, :3] - t_world_rest[tname["L_Foot"]][3, :3])
    scale = leg_t / leg_s
    cj = {j: i for i, j in enumerate(clip["joints"])}
    out_t, out_r, out_s = [], [], []
    for f in range(len(clip["t"])):
        local = []
        for i, j in enumerate(sj):
            k = cj.get(j)
            if k is None:
                local.append(srest[i])
            else:
                local.append(_mat(_quat_to_rot(clip["r"][f][k]), clip["t"][f][k], clip["s"][f][k]))
        s_world = _world(local, sp)
        t_world = [None] * len(tj)
        t_local = [None] * len(tj)
        mapped = {MAP[n]: sname[n] for n in MAP if n in sname}
        for i, j in enumerate(tj):
            n = j.split("/")[-1]
            par = t_world[tp[i]] if tp[i] >= 0 else np.eye(4)
            if n in mapped:
                si = mapped[n]
                D = _rot(s_world_rest[si]).T @ _rot(s_world[si])       # rest^-1 * now (row-vector world delta)
                D_t = A.T @ D @ A
                g_rot = _rot(t_world_rest[i]) @ D_t
                if n == "Hip":   # pelvis translation offset from rest, scaled and aligned
                    off = (s_world[si][3, :3] - s_world_rest[si][3, :3]) @ A * scale
                    g_pos = t_world_rest[i][3, :3] + off
                else:
                    g_pos = (trest[i] @ par)[3, :3]
                g = _mat(g_rot, g_pos)
                loc = g @ np.linalg.inv(par)
                loc[:3, :3] = _rot(loc)
            else:
                loc = trest[i]
                g = loc @ par
            t_world[i], t_local[i] = g, loc
        out_t.append([m[3, :3] for m in t_local])
        out_r.append([_rot_to_quat(_rot(m)) for m in t_local])
        out_s.append([np.ones(3) for _ in t_local])
    return {"joints": np.array(tj), "t": np.array(out_t, np.float32), "r": np.array(out_r, np.float32),
            "s": np.array(out_s, np.float32), "fps": clip["fps"], "world_rest": t_world_rest}


def fk_points(res, f):
    tj = list(res["joints"])
    tp = _parents(tj)
    local = [_mat(_quat_to_rot(res["r"][f][i]), res["t"][f][i]) for i in range(len(tj))]
    w = _world(local, tp)
    return np.array([m[3, :3] for m in w]), tp


if __name__ == "__main__":
    d = Path(sys.argv[1])
    out = Path(__file__).parent / "anim"
    out.mkdir(exist_ok=True)
    src = _skeleton(d / "biped.usd")
    clips = {"walk": load_clip(d / "walk.usd"), "idle": load_clip(d / "idle.usd")}
    for cf in sorted(d.glob("ch_*.usd")):
        name = cf.stem[3:]
        tgt = _skeleton(cf)
        for cn, clip in clips.items():
            r = retarget(src, clip, tgt)
            np.savez_compressed(out / f"{name}_{cn}.npz", joints=r["joints"], t=r["t"], r=r["r"], s=r["s"], fps=r["fps"])
            print("baked", name, cn, r["t"].shape)
        if "--plot" in sys.argv and name == "F_Business_02":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            r = retarget(src, clips["walk"], tgt)
            fig, axes = plt.subplots(1, 4, figsize=(14, 5))
            for ax, f in zip(axes, (0, 20, 40, 60)):
                P, par = fk_points(r, f)
                for i, p in enumerate(par):
                    if p >= 0:
                        ax.plot([P[i, 0], P[p, 0]], [P[i, 2], P[p, 2]], "k-", lw=1)   # side view: x vs z
                ax.set_aspect("equal"); ax.set_title(f"frame {f} (x-z)")
            fig.savefig(out / "retarget_check.png", dpi=80)
