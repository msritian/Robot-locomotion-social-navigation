"""Episode metrics (Section 10). Input: the log from pf.eval.episode.run_episode."""
from __future__ import annotations

import numpy as np

from pf.world.robot import wrap


def _runs(mask):
    """List of (start, end_exclusive) for True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def turn_events(heading, target, dt, min_turn_deg=60.0, window_s=2.0, min_speed=0.1):
    """Start indices of target turns: heading changes > min_turn_deg within window_s while walking."""
    w = int(round(window_s / dt))
    sp = np.r_[0.0, np.hypot(*np.diff(target, axis=0).T) / dt]
    ev, k = [], 0
    while k < len(heading) - w:
        if sp[k:k + w].mean() > min_speed and abs(wrap(heading[k + w] - heading[k])) > np.deg2rad(min_turn_deg):
            rate = np.abs(wrap(np.diff(heading[k:k + w + 1])))
            ev.append(k + int(np.argmax(rate > np.deg2rad(5))))
            k += w
        else:
            k += 1
    return ev


def episode_metrics(log, band=(1.0, 3.0), person_collide=0.45, personal_space=0.5):
    dt = log["dt"]
    ti = log["target_index"]
    robot, people = log["robot"], log["people"]
    n = len(robot)
    dist_t = np.hypot(*(log["target"] - robot[:, :2]).T)
    in_band = (dist_t >= band[0]) & (dist_t <= band[1])
    tracked = log["in_fov"] & in_band
    m = {}
    m["tracking_rate"] = float(tracked.mean())
    # person collisions (onsets) and personal space (non-target people)
    d_all = np.linalg.norm(people - robot[:, None, :2], axis=2)
    d_coll = log.get("robot_person_d", d_all)       # measured right after the robot's move
    coll = (np.minimum(d_coll, d_all) < person_collide).any(axis=1)
    m["person_collisions"] = len(_runs(coll))
    others = np.delete(d_all, ti, axis=1)
    if others.shape[1]:
        m["personal_space_compliance"] = float((others >= personal_space).all(axis=1).mean())
        m["min_dist_other"] = float(others.min())
    else:
        m["personal_space_compliance"] = 1.0
        m["min_dist_other"] = np.nan
    m["min_dist_target"] = float(dist_t.min())
    m["obstacle_collisions"] = len(_runs(log["collided"]))
    # lost events: true target not visible for > 1 s
    one_s = int(round(1.0 / dt))
    lost = [(a, b) for a, b in _runs(~log["visible"]) if b - a > one_s]
    m["lost_events"] = len(lost)
    sel = log["sel_gt"]
    rec_times, recovered = [], 0
    for a, _b in lost:
        # a lost event starts once the target has been out of view for 1 s; recovery = the true target is
        # selected again within 15 s of that; time is measured from when the target left the view
        start = a + one_s
        hit = np.flatnonzero(sel[start:min(n, start + int(round(15.0 / dt)))] == ti)
        if len(hit):
            recovered += 1
            rec_times.append((start + hit[0] - a) * dt)
    m["recovery_rate"] = recovered / len(lost) if lost else np.nan
    m["recovery_time"] = float(np.mean(rec_times)) if rec_times else np.nan
    # wrong-person switches: selected track belongs to a non-target for > 1 s continuously
    wrong = (sel != ti) & (sel != -2)
    m["wrong_switches"] = sum(1 for a, b in _runs(wrong) if b - a > one_s)
    m["wrong_frac"] = float(wrong.mean())
    # following the true target at the end (last selected track within the final 2 s is the target)
    tail = sel[-int(round(2.0 / dt)):]
    tail = tail[tail != -2]
    m["following_at_end"] = bool(len(tail) and tail[-1] == ti)
    m["success"] = bool(m["tracking_rate"] >= 0.8 and m["person_collisions"] == 0 and m["following_at_end"])
    m["smoothness"] = float(np.abs(np.diff(log["cmd"], axis=0)).sum(axis=1).mean())
    # turn lag: after each target turn, time until the target is back inside FOV and band
    lags = []
    for k in turn_events(log["heading_target"], log["target"], dt):
        # 0 if the target stays in FOV and band for 3 s after the turn starts; otherwise the time from the
        # turn start until it is back in FOV and band
        lost_after = np.flatnonzero(~tracked[k:k + int(round(3.0 / dt))])
        if len(lost_after) == 0:
            lags.append(0.0)
            continue
        j = k + lost_after[0]
        back = np.flatnonzero(tracked[j:])
        lags.append((j - k + (back[0] if len(back) else n - j)) * dt)
    m["turn_lag"] = float(np.mean(lags)) if lags else np.nan
    m["n_turns"] = len(lags)
    m["n_swaps_injected"] = int(log["n_swaps"])
    return m
