"""EVT-Bench settings for our brain (shared by the Habitat agent and the offline replay)."""

CTRL_FREQ = 40.0
LIN, LAT, ANG = 15.0, 10.0, 6.28          # benchmark speeds for action = 1 (config agent_1_base_velocity)
HUMAN_LIN = 10.0                          # main human oracle nav lin_speed
# One benchmark step moves a person up to HUMAN_LIN / CTRL_FREQ = 0.25 m. We call one step 0.25 s so people
# walk at ~1 m/s, a normal pace. Only the ratio matters to the brain; the robot gets the same limits as the VLA.
STEP_S = 0.25
VX_MAX = LIN / CTRL_FREQ / STEP_S         # 1.5 m/s
VY_MAX = LAT / CTRL_FREQ / STEP_S         # 1.0 m/s
WZ_MAX = ANG / CTRL_FREQ / STEP_S         # 0.63 rad/s
HUMAN_IDS = (1000, 1100)                  # other humans' panoptic ids (evt_bench additional_sensor)
EVT_OVERRIDES = {
    "sim": {"dt": STEP_S},
    "robot": {"vx_range": [-0.5, VX_MAX], "vy_range": [-VY_MAX, VY_MAX], "wz_range": [-WZ_MAX, WZ_MAX],
              "acc_lin": 4.0, "acc_ang": 4.0, "radius": 0.30, "fov_deg": 90.0, "cam_range": [0.3, 12.0],
              "tau": 0.05, "latency": 0.0},           # the benchmark's base follows commands instantly
    # the benchmark counts "following" only while the target fills >3000 px (about <2.5 m away): follow a bit closer;
    # its targets often walk toward the robot at the start: then hold the distance instead of circling behind them
    "follow": {"distance": 1.3, "hold_off_when_approached": True},
}
# tuning variants (tuned on the every-10th MP3D STT subset only; that subset is excluded from reported results)
VARIANTS = {
    "base": {},
    "head2": {"dwa": {"w_heading": 2.0}},
    "head2_d1": {"dwa": {"w_heading": 2.0}, "follow": {"distance": 1.0}},
    "head3_g05": {"dwa": {"w_heading": 3.0, "w_goal": 0.5}},
    # 4 Hz updates, abrupt human starts/stops, mask-based positions: wider tracker gate
    "trk4": {"tracker": {"q_acc": 4.0, "min_meas_sigma": 0.15}},
    "trk8": {"tracker": {"q_acc": 8.0, "min_meas_sigma": 0.25}},
    # target walking at the robot: step aside instead of holding / backing up (backing is slower than the human)
    "giveway": {"follow": {"give_way": True}},
    # no sample keeps 0.5 m from a person: take the wall-safe one with the most distance (default: stop + rotate)
    "evade": {"dwa": {"evade_fallback": True}},
    "evade_gw": {"dwa": {"evade_fallback": True}, "follow": {"give_way": True}},
    # + goal cost = path length on the robot's own obstacle map (gets around walls / through doorways)
    "evgw_path": {"dwa": {"evade_fallback": True, "path_cost": True}, "follow": {"give_way": True}},
    # + rollouts hold the command 0.5 s then brake (re-planned every 0.25 s; 1.5 s straight lines hit walls indoors)
    # brake rollouts for walls only; people judged on the held command over the full horizon
    "evgw_bw": {"dwa": {"evade_fallback": True, "brake_after_s": 0.5, "brake_walls_only": True}, "follow": {"give_way": True}},
    "evgw_pbw": {"dwa": {"evade_fallback": True, "path_cost": True, "brake_after_s": 0.5, "brake_walls_only": True},
                 "follow": {"give_way": True}},
    "evgw_b": {"dwa": {"evade_fallback": True, "brake_after_s": 0.5}, "follow": {"give_way": True}},
    "evgw_pb": {"dwa": {"evade_fallback": True, "path_cost": True, "brake_after_s": 0.5}, "follow": {"give_way": True}},
}
