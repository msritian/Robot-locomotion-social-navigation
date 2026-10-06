"""Animated human characters for Part D (PROJECT_SPEC 13.4 option 1/2): Isaac Sim's rigged people
(Isaac/People/Characters, NVIDIA limited-use assets) moved along the Stage C people trajectories, playing the
in-place walk cycle (Isaac/People/Animations/stand_walk_loop_in_place) at a rate matched to their ground speed,
or the idle loop when standing. Pure USD animation (no physics); robot-person contact is checked by distance.

Each character owns a SkelAnimation prim whose joint arrays are overwritten every frame with a sample of the
source clip at that character's own phase, so people walk out of sync and at their own speeds.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

# characters with baked retargeted clips (isaac_follow/retarget.py -> isaac_follow/anim/<name>_{walk,idle}.npz)
CHARACTERS = ["F_Business_02", "M_Medical_01", "F_Medical_01", "male_adult_construction_01_new",
              "female_adult_police_01_new"]
TARGET_CHARACTER = "F_Business_02"      # look-alikes use the same model
WALK_CLIP = "Isaac/People/Animations/stand_walk_loop_in_place.skelanim.usd"
IDLE_CLIP = "Isaac/People/Animations/stand_idle_loop.skelanim.usd"
CLIP_SPEED = 1.08                         # m/s, from stand_walk_loop (2.89 m per 80 frames at 30 fps)
YAW_OFFSET = np.pi / 2                    # character model faces -y (checked in the warehouse render)
ANIM_DIR = __import__("pathlib").Path(__file__).parent / "anim"


def _load_clip(url):
    st = Usd.Stage.Open(url)
    anim = UsdSkel.Animation(st.GetDefaultPrim())
    t0, t1 = st.GetStartTimeCode(), st.GetEndTimeCode()
    times = np.arange(t0, t1 + 1e-6)
    clip = {
        "joints": anim.GetJointsAttr().Get(),
        "t": [anim.GetTranslationsAttr().Get(t) for t in times],
        "r": [anim.GetRotationsAttr().Get(t) for t in times],
        "s": [anim.GetScalesAttr().Get(t) for t in times],
        "fps": st.GetTimeCodesPerSecond(), "n": len(times),
    }
    return clip


class CharacterCrowd:
    def __init__(self, stage, base_url, n, target_index, lookalike_ids=(), seed=0, ring=True):
        self.stage = stage
        self.n = n
        rng = np.random.default_rng(seed)
        self.clips = {}
        for c in CHARACTERS:
            self.clips[c] = {k: self._baked(c, k) for k in ("walk", "idle")}
        self.phase = rng.uniform(0, 1, n)
        self.xf, self.anim, self.names = [], [], []
        others = [c for c in CHARACTERS if c != TARGET_CHARACTER]
        for i in range(n):
            name = TARGET_CHARACTER if (i == target_index or i in lookalike_ids) else others[(i + seed) % len(others)]
            root = UsdGeom.Xform.Define(stage, f"/World/People/P{i:02d}")
            ops = (root.AddTranslateOp(), root.AddRotateZOp())
            self.xf.append(ops)
            char = stage.DefinePrim(f"/World/People/P{i:02d}/Char")
            char.GetReferences().AddReference(f"{base_url}/Isaac/People/Characters/{name}/{name}.usd")
            skel = next((p for p in Usd.PrimRange(char) if p.IsA(UsdSkel.Skeleton)), None)
            anim = UsdSkel.Animation.Define(stage, f"/World/People/P{i:02d}/Anim")
            anim.CreateJointsAttr().Set(self.clips[name]["walk"]["joints"])
            self.names.append(name)
            if skel is not None:
                UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
            self.anim.append(anim)
            if ring and i == target_index:     # thin red ring at the target's feet (clear in chase/top views)
                c = UsdGeom.Cylinder.Define(stage, f"/World/People/P{i:02d}/Ring")
                c.CreateRadiusAttr(0.45)
                c.CreateHeightAttr(0.02)
                c.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.05, 0.05)])
                c.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.012))

    def update(self, pos, heading, speed, dt):
        for i in range(self.n):
            tr, rz = self.xf[i]
            tr.Set(Gf.Vec3d(float(pos[i, 0]), float(pos[i, 1]), 0.0))
            rz.Set(float(np.rad2deg(heading[i] + YAW_OFFSET)))
            moving = speed[i] > 0.08
            clip = self.clips[self.names[i]]["walk" if moving else "idle"]
            dur = clip["n"] / clip["fps"]
            rate = (speed[i] / CLIP_SPEED) if moving else 1.0
            self.phase[i] = (self.phase[i] + rate * dt / dur) % 1.0
            k = int(self.phase[i] * clip["n"]) % clip["n"]
            a = self.anim[i]
            a.GetTranslationsAttr().Set(clip["t"][k])
            a.GetRotationsAttr().Set(clip["r"][k])
            a.GetScalesAttr().Set(clip["s"][k])

    @staticmethod
    def _baked(name, kind):
        d = np.load(ANIM_DIR / f"{name}_{kind}.npz")
        n = len(d["t"])
        return {"joints": Vt.TokenArray([str(j) for j in d["joints"]]), "n": n, "fps": float(d["fps"]),
                "t": [Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(d["t"][f])) for f in range(n)],
                "r": [Vt.QuatfArray([Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))
                                     for q in d["r"][f]]) for f in range(n)],
                "s": [Vt.Vec3hArray.FromNumpy(np.ascontiguousarray(d["s"][f].astype(np.float16))) for f in range(n)]}


def add_visual_environment(stage, url, offset, prim_path="/World/Environment"):
    """Reference a photoreal Isaac environment AFTER the simulation started and strip all physics schemas from it
    (as session-layer edits), so PhysX never parses its ~700k collision triangles / rigid props."""
    from pxr import UsdPhysics
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*offset))
    root.GetPrim().GetReferences().AddReference(url)
    stage.Load(root.GetPath())
    n = 0
    apis = (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI)
    for prim in Usd.PrimRange(root.GetPrim(), Usd.TraverseInstanceProxies()):
        if prim.IsInstanceProxy():
            continue
        for api in apis:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                n += 1
        for name in ("physics:collisionEnabled", "physics:rigidBodyEnabled"):
            a = prim.GetAttribute(name)
            if a and a.IsValid():
                a.Set(False)
        if prim.IsInstance():               # instanced props: make the instance prim itself non-physical
            prim.SetInstanceable(False)
    return n
