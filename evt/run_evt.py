"""Run one EVT-Bench chunk for one agent, on the MP3D episodes only (the scenes we have).

Run from the OmTrackVLA checkout (it holds habitat-lab, evt_bench and data/), with this repo on PYTHONPATH:
    python run_evt.py --agent ours --task stt --split-num 40 --split-id 0 --save-path out/ours/stt
    python run_evt.py --agent omtrackvla ...      # OmTrackVLA's own agent (HF_MODEL_ID / HF_MODEL_DIR set)
    python run_evt.py --probe                     # plumbing check: sensors, frames, speeds per step
Every agent gets the same episodes: MP3D episodes in file order, chunk i = episodes[i::split_num].
"""
import argparse
import os
import random
import time

import numpy as np

import habitat
from habitat.datasets import make_dataset
import evt_bench  # noqa: F401  (registers the tracking task, sensors and actions)

CFG = "habitat-lab/habitat/config/benchmark/nav/track/track_infer_{}.yaml"


def episodes(config, scenes, every, exclude_every=0, scene_ids=None):
    """exclude_every=10 drops the tuning subset (= every 10th episode, the pilots' EVERY=10) for the reported runs."""
    ds = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    eps = [e for e in ds.episodes if f"/{scenes}/" in e.scene_id or e.scene_id.startswith(f"{scenes}/")]
    if scene_ids:
        eps = [e for e in eps if e.scene_id.split("/")[-1].split(".")[0] in scene_ids]
    if exclude_every:
        eps = [e for i, e in enumerate(eps) if i % exclude_every]
    ds.episodes = eps[::every]
    return ds


def probe(config, ds):
    from evt.ours_agent import DEPTH, PAN, RGB, LIGHTS, pose2d, EVTPerception, OursAgent, align_depth_camera
    ds.episodes = ds.episodes[:1]
    with habitat.TrackEnv(config=config, dataset=ds) as env:
        sim = env.sim
        env.reset()
        sim.set_light_setup(LIGHTS)
        align_depth_camera(sim)
        ep = env.current_episode
        print("[probe] episode", ep.episode_id, ep.scene_id, "main id", ep.info.get("main_human_semantic_id"),
              "instruction", ep.info.get("instruction"))
        obs = sim.get_sensor_observations()
        for k, v in obs.items():
            a = np.asarray(v)
            print(f"[probe] obs {k}: {a.shape} {a.dtype} min={a.min():.3f} max={a.max():.3f}")
        det = env.task._get_observations(ep)
        print("[probe] task obs keys", list(det.keys()))
        robot = sim.agents_mgr[1].articulated_agent
        human = sim.agents_mgr[0].articulated_agent
        per = EVTPerception(OursAgent("/tmp/probe").cfg, *np.asarray(obs[DEPTH]).shape[:2])
        idle = ("agent_0_humanoid_navigate_action", "agent_1_base_velocity", "agent_2_oracle_nav_randcoord_action_obstacle",
                "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle",
                "agent_5_oracle_nav_randcoord_action_obstacle")
        main_id = int(ep.info["main_human_semantic_id"])

        def state():
            o = sim.get_sensor_observations()
            cam, cy = pose2d(sim._sensors[DEPTH]._sensor_object.node.absolute_transformation())
            hb = np.array([human.base_pos[0], -human.base_pos[2]])
            rb = np.array([robot.base_pos[0], -robot.base_pos[2]])
            return o, cam, cy, hb, rb
        for name, a in (("still", [0, 0, 0]), ("fwd", [1, 0, 0]), ("left", [0, 1, 0]), ("turn", [0, 0, 1])):
            o, cam0, _, h0, r0 = state()
            t0 = time.time()
            for _ in range(8):
                env.step({"action": idle, "action_args": {"agent_1_base_vel": a}})
            dt = (time.time() - t0) / 8
            o, cam1, cy, h1, r1 = state()
            print(f"[probe] {name:5s} a={a}: robot base moved {np.round(r1 - r0, 3)} /8 steps, cam yaw {cam0[2]:.3f}->{cam1[2]:.3f}, "
                  f"cam-base offset {np.round(cam1[:2] - r1, 3)}, human moved {np.linalg.norm(h1 - h0) / 8:.3f} m/step, {dt * 1e3:.0f} ms/step")
            d = np.asarray(o[DEPTH])[..., 0] if np.asarray(o[DEPTH]).ndim == 3 else np.asarray(o[DEPTH])
            p = np.asarray(o[PAN])[..., 0] if np.asarray(o[PAN]).ndim == 3 else np.asarray(o[PAN])
            dets = per.people(np.asarray(o[RGB])[..., :3], d, p, main_id)
            c, s = np.cos(cam1[2]), np.sin(cam1[2])
            for q, g in zip(dets.pos_robot, dets.gt_id):
                w = cam1[:2] + np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]])
                tag = "TARGET" if g == main_id else str(g)
                err = np.linalg.norm(w - h1) if g == main_id else float("nan")
                print(f"[probe]   det {tag}: robot-frame {np.round(q, 2)} world {np.round(w, 2)} (target true {np.round(h1, 2)}, err {err:.2f} m)")
            print(f"[probe]   rays (m): {np.round(per.rays(d, p, cy - float(sim.pathfinder.snap_point(robot.base_pos)[1]), main_id), 1)}")


def omtrackvla_preflight(trained_agent):
    """OmTrackVLA swallows load errors (no vision encoder / planner -> silent garbage actions). Use local weights and
    check that every model loads and that one frame gives an action before running episodes."""
    import functools
    import os
    import torch
    siglip = os.environ.get("SIGLIP_MODEL_PATH")
    if siglip:
        trained_agent.VisionCacheConfig = functools.partial(trained_agent.VisionCacheConfig, siglip_model_name=siglip)
    for k in ("HF_MODEL_DIR", "DINOV3_MODEL_PATH", "SIGLIP_MODEL_PATH"):
        assert os.path.isdir(os.environ.get(k, "")), f"{k} not set / missing"
    a = trained_agent.GTBBoxAgent("/tmp/omt_preflight")
    a.reset()
    assert a._ensure_vision_cache() is not None, "vision encoders (DINOv3 / SigLIP) failed to load"
    a._resolve_planner_ckpt_once()
    assert a._hf_model_dir, "HF_MODEL_DIR not accepted"
    a._init_planner_model()
    assert a.planner_model is not None, "planner failed to load"
    vc, vf = a._encode_frame_tokens(np.zeros((384, 384, 3), np.uint8))
    assert vc is not None and vf is not None, "frame encoding failed"
    act = a._planner_action(np.full((384, 384, 3), 128, np.uint8), "follow the person")
    assert act is not None, "planner gave no action"
    print(f"[preflight] OmTrackVLA ok: dino={a._vision_cache.cfg.dino_model_name} siglip={a._vision_cache.cfg.siglip_model_name} "
          f"llm={a.planner_model.config.llm_name} cuda={torch.cuda.is_available()} action={act}", flush=True)
    a.reset()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="ours", choices=["ours", "omtrackvla", "hybrid"])
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="train = development episodes (MP3D train scenes, disjoint from the 14 val scenes)")
    ap.add_argument("--scene-ids", nargs="+", default=None, help="only these scenes (dev: the MP3D train scenes we fetched)")
    ap.add_argument("--exclude-every", type=int, default=0, help="drop every k-th episode (10 = the tuning subset)")
    ap.add_argument("--shield", default="short", choices=["short", "short_tgt", "evade", "dwa", "none"], help="hybrid: safety check of VLA commands")
    ap.add_argument("--takeover", default="distractor", choices=["distractor", "distractor_search", "all", "none"],
                    help="hybrid: when our controller drives")
    ap.add_argument("--no-shield", action="store_true", help="= --shield none")
    ap.add_argument("--no-takeover", action="store_true", help="= --takeover none")
    ap.add_argument("--text-log", action="store_true", help="hybrid: compute and log SigLIP text scores")
    ap.add_argument("--agree-min", type=int, default=10, help="hybrid: distractor takeover after this many agreeing steps")
    ap.add_argument("--app-conf", type=float, default=0.85, help="hybrid: memory confidence needed for distractor takeover")
    ap.add_argument("--shield-walls", action="store_true", help="hybrid: the shield also checks walls (real robot)")
    ap.add_argument("--init", default="vla", choices=["vla", "text", "first_frame"], help="hybrid: how our memory learns the target")
    ap.add_argument("--task", default="stt", choices=["stt", "dt", "at"])
    ap.add_argument("--scenes", default="mp3d")
    ap.add_argument("--every", type=int, default=1, help="keep every k-th episode (subset)")
    ap.add_argument("--split-num", type=int, default=1)
    ap.add_argument("--split-id", type=int, default=0)
    ap.add_argument("--save-path", default="out")
    ap.add_argument("--controller", default="C1")
    ap.add_argument("--memory", default="M2")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    from evt.ours_agent import depth_sensor_opts
    opts = list(args.opts) + [f"habitat.dataset.split={args.split}"]
    opts += depth_sensor_opts() if args.agent in ("ours", "hybrid") or args.probe else []
    config = habitat.get_config(CFG.format(args.task), opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)
    ds = episodes(config, args.scenes, args.every, args.exclude_every, args.scene_ids)
    print(f"[run_evt] {args.task}: {len(ds.episodes)} {args.scenes} episodes (every {args.every}, excluding every {args.exclude_every})")
    if args.probe:
        return probe(config, ds)
    ds.episodes = ds.episodes[args.split_id::args.split_num]
    print(f"[run_evt] chunk {args.split_id}/{args.split_num}: {len(ds.episodes)} episodes, agent {args.agent}")
    if args.agent == "ours":
        from evt.ours_agent import VARIANTS, OursAgent, evaluate_agent
        agent = OursAgent(args.save_path, args.controller, args.memory, cfg_over=VARIANTS[args.variant])
        evaluate_agent(config, ds, args.save_path, agent)
    elif args.agent == "hybrid":
        import trained_agent
        from evt.hybrid_agent import HybridAgent
        from evt.ours_agent import VARIANTS, evaluate_agent
        vla = omtrackvla_preflight(trained_agent)
        grounder = None
        if args.init == "text" or args.text_log:
            from evt.grounding import TextGrounder
            grounder = TextGrounder(os.environ["SIGLIP_MODEL_PATH"])
            grounder.set_instruction("Follow the man in a red shirt.")
            assert grounder.scores(np.zeros((64, 64, 3), np.uint8), [(0, 64, 0, 64)]).shape == (1,)
            print("[preflight] SigLIP text grounding ok", flush=True)
        agent = HybridAgent(args.save_path, vla, shield="none" if args.no_shield else args.shield,
                            takeover="none" if args.no_takeover else args.takeover, init=args.init,
                            shield_walls=args.shield_walls, agree_min=args.agree_min, app_conf=args.app_conf,
                            grounder=grounder, cfg_over=VARIANTS[args.variant])
        print(f"[run_evt] hybrid shield={agent.shield} takeover={agent.takeover} init={agent.init_mode} agree_min={agent.agree_min} "
              f"app_conf={agent.app_conf} variant={args.variant}", flush=True)
        evaluate_agent(config, ds, args.save_path, agent)
    else:
        import trained_agent
        omtrackvla_preflight(trained_agent)
        trained_agent.evaluate_agent(config, ds, args.save_path)


if __name__ == "__main__":
    main()
