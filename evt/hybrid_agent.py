"""Hybrid follower for EVT-Bench (direction C): OmTrackVLA drives; our modules make it safe and recover the target.

Every step both run on the same observation:
  VLA  (RGB + instruction) -> command + predicted waypoints (robot frame)
  ours (panoptic person masks without identity + depth) -> tracker -> target memory (M2) -> planner / search

Components (each can be switched off for ablations):
  shield    "short": the VLA command is kept unless it brings the robot within 0.5 m of a tracked person in the next
            0.5 s; then the same command is tried slower (down to a stop).  "dwa": our planner's full check (1.5 s,
            0.5 m around people; walls from depth only with shield_walls). EVT-Bench's base slides along walls on the
            navmesh and only people end an episode, so wall checks there only stop the VLA in narrow corridors.
            "evade": "short" against EVERY tracked person (own all-people tracker, so also before our memory has
            learned the target); if even stopping is unsafe (someone walks into the robot), our planner picks the
            move that keeps the most distance (walls ignored, see above).
            "short_tgt": "short" against the target only (pilots: "short" against everyone lowered DT success by stopping
            for distractors; "dwa" lowered it more).
  takeover  "distractor_search": "distractor", plus our search drives while our memory has lost the target and the
            VLA heads for someone who does not look like it (never when the VLA heads for nobody: that is what hurt STT).
            "distractor": our controller drives while the VLA heads for a different person than the one our memory
            confidently sees (pilot precision 90%).  "all": also while our memory has lost the target and the person
            the VLA heads for (if any) does not look like the target; pilots: our search drove away from targets
            the VLA would have re-found (STT success 88% -> 75%), so "distractor" is the default.
  init      how our memory learns the target: "vla" = the person the VLA heads for over the first steps (same inputs
            as the VLA alone), "text" = the person whose crop best matches the instruction (SigLIP, evt/grounding.py;
            also RGB + instruction only), "first_frame" = the person in view at the start (our agent's protocol).
The person "the VLA heads for" = the detection whose bearing is closest to the VLA trajectory's end point.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np

from evt.evt_cfg import STEP_S, VX_MAX, VY_MAX, WZ_MAX
from evt.ours_agent import DEPTH, PAN, RGB, EVTPerception, OursAgent, pose2d
from pf.perception.tracker import Tracker
from pf.world.robot import integrate_pose

MAX = np.array([VX_MAX, VY_MAX, WZ_MAX])
IMPLIED_DEG = 20.0      # a person counts as the VLA's heading target within this bearing of the trajectory end
INIT_STEPS = 6          # "vla" init: same person implied this many steps in a row
AGREE_MIN = 10          # distractor takeover only after memory and VLA agreed on the target this many steps
SEARCH_AFTER_S = 2.0    # takeover="all": the VLA gets this long to re-find a lost target before our search drives
SHIELD_SCALES = (1.0, 0.66, 0.33)   # unsafe VLA command: try it slower (same direction) before another command
SHORT_STEPS = 2         # shield="short": look-ahead in steps (0.5 s)
APP_SAME = 0.6          # appearance score below this = not the target (true target median 0.94, others' 95th pct 0.54)
APP_CONF = 0.85         # memory's own selection is confident above this
TEXT_STEPS = 3          # "text" init: frames with the same best-matching person (by running mean logit)
TEXT_MARGIN = 1.0       # "text" init: lead over the runner-up's mean logit (when someone else was seen)


class HybridAgent(OursAgent):
    def __init__(self, result_path, vla, shield="short", takeover="distractor", init="vla", shield_walls=False,
                 agree_min=AGREE_MIN, app_conf=APP_CONF, grounder=None, cfg_over=None, **kw):
        super().__init__(result_path, cfg_over=cfg_over, **kw)
        assert shield in ("none", "short", "short_tgt", "evade", "dwa")
        assert takeover in ("none", "distractor", "distractor_search", "all")
        self.vla, self.shield, self.takeover, self.init_mode, self.shield_walls = vla, shield, takeover, init, shield_walls
        self.agree_min, self.app_conf, self.grounder = agree_min, app_conf, grounder
        assert init != "text" or grounder is not None

    def start(self, episode, main_id=None):
        super().start(episode, main_id)
        self.instruction = (episode.info or {}).get("instruction")
        self.vla.rgb_list = []
        self.vla.reset()
        self.implied_hist = []
        self.modes = []
        self.agree = 0
        self.all_trk = Tracker(self.cfg["tracker"], STEP_S, 1.0, np.random.default_rng(0))   # every person (shield)
        self.all_tracks = []
        self.text_sum, self.text_n, self.text_lead = {}, {}, []
        if self.grounder is not None:
            self.grounder.set_instruction(self.instruction)

    # --------------------------------------------------------------------------------------------- helpers
    def _vla(self, obs):
        with contextlib.redirect_stdout(io.StringIO()):          # the VLA prints arrays every step
            a = self.vla.act(obs, None, self.ep_tag, self.instruction)
        self.vla.rgb_list = []
        traj = self.vla._last_predicted_traj
        return (None if a is None else np.asarray(a, float)), (None if traj is None else np.asarray(traj, float))

    @staticmethod
    def _implied(det, traj):
        """Index of the detection the VLA trajectory heads for, or None."""
        if traj is None or len(det.gt_id) == 0:
            return None
        end = traj[-1, :2]
        if np.hypot(*end) < 0.05:
            return None
        b = np.arctan2(end[1], end[0])
        bd = np.abs(np.angle(np.exp(1j * (np.arctan2(det.pos_robot[:, 1], det.pos_robot[:, 0]) - b))))
        i = int(np.argmin(bd))
        return i if np.degrees(bd[i]) <= IMPLIED_DEG else None

    def _all_people(self):
        return [(np.asarray(t["pos_world_est"]), np.asarray(t["vel_world_est"])) for t in self.all_tracks
                if t["time_since_seen_s"] <= 0.5]

    def _short_safe(self, odom, cmd, target_only=False, ppl=None):
        """No tracked person (target included, constant velocity) within person_collide in the next SHORT_STEPS."""
        if ppl is None:
            ppl = [] if target_only else list(self.brain.others)
            ppl += [self.brain.target_pv] if self.brain.target_pv is not None else []
        if not ppl:
            return True
        lim, dt = self.cfg["dwa"]["person_collide"], self.brain.dt
        pose = np.asarray(odom, float)
        for k in range(1, SHORT_STEPS + 1):
            pose = integrate_pose(pose, np.asarray(cmd, float), dt)
            if any(np.hypot(*(pose[:2] - (np.asarray(p) + k * dt * np.asarray(v)))) < lim for p, v in ppl):
                return False
        return True

    # --------------------------------------------------------------------------------------------- step
    def act(self, sim, obs):
        rgb, depth, pan = obs[RGB][..., :3], obs[DEPTH], obs[PAN]
        depth = depth[..., 0] if depth.ndim == 3 else depth
        pan = pan[..., 0] if pan.ndim == 3 else pan
        if self.per is None:
            self.per = EVTPerception(self.cfg, *depth.shape)
        odom, cam_y = pose2d(sim._sensors[DEPTH]._sensor_object.node.absolute_transformation())
        floor_y = float(sim.pathfinder.snap_point(sim.agents_mgr[1].articulated_agent.base_pos)[1])
        a_vla, traj = self._vla(obs)

        det = self.per.people(rgb, depth, pan, self.main_id)
        imp = self._implied(det, traj)
        tx = self.grounder.scores(rgb, self.per.last_boxes) if (self.grounder is not None and len(det.gt_id)) \
            else np.zeros(len(det.gt_id))
        mem = self.brain.memory
        if not mem.initialized:
            # our memory only sees the person chosen by the init protocol (gt_id below is just a handle to a detection)
            if self.init_mode == "first_frame":
                keep = det.gt_id == self.main_id
            elif self.init_mode == "text":
                keep = np.zeros(len(det.gt_id), bool)
                for g, v in zip(det.gt_id, tx):
                    self.text_sum[int(g)] = self.text_sum.get(int(g), 0.0) + float(v)
                    self.text_n[int(g)] = self.text_n.get(int(g), 0) + 1
                if len(det.gt_id):
                    mean = {g: self.text_sum[g] / self.text_n[g] for g in self.text_sum}
                    rank = sorted(mean, key=mean.get, reverse=True)
                    lead = rank[0] if (len(rank) == 1 or mean[rank[0]] - mean[rank[1]] > TEXT_MARGIN) else None
                    self.text_lead.append(lead if lead in det.gt_id else None)
                    h = self.text_lead[-TEXT_STEPS:]
                    if len(h) == TEXT_STEPS and h[0] is not None and all(x == h[0] for x in h):
                        keep = det.gt_id == h[0]
            else:
                self.implied_hist.append(None if imp is None else int(det.gt_id[imp]))
                h = self.implied_hist[-INIT_STEPS:]
                stable = len(h) == INIT_STEPS and h[0] is not None and all(x == h[0] for x in h)
                keep = (det.gt_id == h[0]) if stable else np.zeros(len(det.gt_id), bool)
            feed = det.subset(keep) if hasattr(det, "subset") else _subset(det, keep)
        else:
            feed = det
        tracks = self.tracker.update(feed, odom)
        self.all_tracks = self.all_trk.update(det, odom)
        rays = self.per.rays(depth, pan, cam_y - floor_y, self.main_id)
        bobs = {"tracks": tracks, "robot_odom": odom.copy(), "ray_angles": self.per.ray_angles, "depth_rays": rays}
        cmd_ours = self.brain.step(bobs, commit=False)
        if cmd_ours is None or not np.all(np.isfinite(cmd_ours)):
            cmd_ours = np.zeros(3)
        sel = self.brain.selected

        # ---- who drives
        s_imp = float(mem.s_app(det.appearance[imp][None])[0]) if (imp is not None and mem.initialized) else None
        imp_is_sel, sel_det = False, None
        if sel is not None and len(det.gt_id):
            dd = np.hypot(*(det.pos_robot - np.asarray(sel["pos_robot"])).T)
            sel_det = int(np.argmin(dd)) if dd.min() < 0.5 else None
        if imp is not None and sel is not None:
            imp_is_sel = np.hypot(*(det.pos_robot[imp] - np.asarray(sel["pos_robot"]))) < 0.5
        if imp_is_sel:
            self.agree += 1
        mode = "vla"
        if a_vla is None:
            mode = "ours"
        elif self.takeover != "none" and mem.initialized:
            if sel is not None:
                if self.agree >= self.agree_min and imp is not None and not imp_is_sel \
                        and mem.last_scores.get("s_app", 0) > self.app_conf and (s_imp is None or s_imp < APP_SAME):
                    mode = "ours:distractor"         # the VLA heads for someone else while we clearly see the target
            elif self.takeover == "all" and mem.time_since_seen > SEARCH_AFTER_S \
                    and (imp is None or (s_imp is not None and s_imp < APP_SAME)):
                mode = "ours:search"                 # nobody we would accept as the target is where the VLA goes
            elif self.takeover == "distractor_search" and mem.time_since_seen > SEARCH_AFTER_S \
                    and s_imp is not None and s_imp < APP_SAME:
                mode = "ours:search"                 # the VLA heads for someone who does not look like the target

        # ---- command
        if mode == "vla":
            out = a_vla.copy()
            if self.shield in ("short", "short_tgt"):
                for f in SHIELD_SCALES + (0.0,):
                    if self._short_safe(odom, f * a_vla * MAX, target_only=self.shield == "short_tgt"):
                        break
                if f != 1.0:
                    out = f * a_vla
                    mode = "vla:shielded"
            elif self.shield == "evade":
                ppl = self._all_people()
                for f in SHIELD_SCALES + (0.0,):
                    if self._short_safe(odom, f * a_vla * MAX, ppl=ppl):
                        break
                else:
                    f = None
                if f is None:                    # even stopping is unsafe: someone walks into us -> get out of the way
                    look = self.brain.target_pv[0] if self.brain.target_pv is not None else None
                    c = self.brain.dwa.plan(odom, None, look, ppl, target=None, pref=a_vla * MAX, walls=False)
                    out, mode = c / MAX, "vla:evade"
                elif f != 1.0:
                    out, mode = f * a_vla, "vla:shielded"
            elif self.shield == "dwa":
                for f in SHIELD_SCALES:
                    pref = f * a_vla * MAX
                    c = self.brain.dwa.plan(odom, None, None, self.brain.others, target=self.brain.target_pv, pref=pref,
                                            walls=self.shield_walls)
                    if np.allclose(c, self.brain.dwa.model.clamp(pref)):
                        break
                if f != 1.0 or not np.allclose(c, self.brain.dwa.model.clamp(pref)):
                    out = c / MAX
                    mode = "vla:shielded"
            sent = out * MAX
        else:
            sent = cmd_ours
            out = cmd_ours / MAX
        self.brain.dwa.commit(sent)
        self.modes.append(mode)

        # ---- logs (METRICS ONLY below: true target position / identity)
        hp = sim.agents_mgr[0].articulated_agent.base_pos
        d = np.array([hp[0], -hp[2]]) - odom[:2]
        c_, s_ = np.cos(odom[2]), np.sin(odom[2])
        rel = [round(float(c_ * d[0] + s_ * d[1]), 2), round(float(-s_ * d[0] + c_ * d[1]), 2)]
        self.raw.append({"odom": odom.copy(), "pos": det.pos_robot.copy(), "gid": det.gt_id.copy(),
                         "app": det.appearance.astype(np.float16), "init": not mem.initialized, "rays": rays.astype(np.float32),
                         "traj": None if traj is None else traj.astype(np.float32), "a_vla": a_vla,
                         "text": tx.astype(np.float32)})
        self.log.append({"mode": mode, "tgt_rel": rel, "state": self.brain.state,
                         "sel_gt": -2 if sel is None else int(sel["_gt_id"]),
                         "vla_gt": -2 if imp is None else int(det.gt_id[imp]), "s_imp": None if s_imp is None else round(s_imp, 3),
                         "scores": {k: round(v, 3) for k, v in mem.last_scores.items()}, "n_det": int(len(det.gt_id)),
                         "t_imp": None if imp is None else round(float(tx[imp]), 2),
                         "t_sel": None if sel_det is None else round(float(tx[sel_det]), 2),
                         "a_vla": None if a_vla is None else [round(float(x), 3) for x in a_vla],
                         "cmd": [round(float(x), 3) for x in out]})
        if self.n_eps < self.save_video_n:
            self.rgb_list.append(rgb.copy())
        return [float(x) for x in out]

    def finish(self, episode, save_dir):
        r = super().finish(episode, save_dir)
        m = self.modes
        r.update({f"mode_{k}": sum(x == k for x in m) for k in ("vla", "vla:shielded", "vla:evade", "ours", "ours:distractor", "ours:search")})
        vg = [l["vla_gt"] for l in self.log if l["vla_gt"] != -2]
        r["vla_steps_on_other"] = sum(g != self.main_id for g in vg)
        return r


def _subset(det, keep):
    from pf.perception.sensor import Detections
    return Detections(pos_robot=det.pos_robot[keep], body_heading_rel=det.body_heading_rel[keep], head_rel=det.head_rel[keep],
                      appearance=det.appearance[keep], gt_id=det.gt_id[keep], meas_sigma=det.meas_sigma[keep])
