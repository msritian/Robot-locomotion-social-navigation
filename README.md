# 🤖🚶 K1 Person Following

**Teaching a small humanoid robot to follow one person through a crowded building, walking on its own legs.**

The robot is a [Booster K1](https://www.boosterobotics.com/) (≈ 1 m tall, 22 joints) in simulation. You point it
at a person; it keeps that person in view at a comfortable distance, weaves around furniture and other people,
doesn't get fooled by someone in a similar shirt, and goes looking if the person disappears around a corner.

> **Status (Oct 2026):** the following "brain" and all 2D experiments work. The K1 walking policy is training on
> the UW–Madison CHTC GPU cluster. Next: the first videos of the walking K1 following a person in realistic,
> crowded indoor scenes.

---

## The idea in one picture

```
   camera view ──► "people as numbers" ──► who is my target? ──► where will they go? ──► where do I step? ──► legs
                   (positions, headings,     (appearance memory     (learned predictor,    (goal behind the     (learned walking
                    look, track IDs)          + motion check)        2 s ahead)             target + obstacle-   policy, 50 Hz)
                                                                                            aware planner)
```

The following brain never looks at raw pixels. A perception layer turns the camera view into **numbers about
people** (where each person is, which way they face, what they look like). Everything downstream reasons about
those numbers. That makes the brain fast to train and test, and easy to move from the simulator to the real robot.

Three ideas are being tested:

| | Idea | In plain words |
|---|---|---|
| 1 | **Predict where the person is going** | Aim where they *will* be, not where they are, so the robot isn't late at every turn. Uses body and head direction as cues (people often look before they turn). |
| 4 | **Never lose the right person** | Remember what the target looks like from several angles, but only add to that memory when it is safe, so a look-alike can't take over. |
| 4b | **Search smartly** | If the person vanishes, go where they were *heading*, not just where they were last seen. |

## Results so far (simulation, held-out test layouts)

Everything below comes from code in this repo, on test seeds never used for tuning. Mean over episodes;
95% bootstrap confidence intervals are in `results/`.

**Predicting the target's path (E1).** Error after 2 s, in meters (lower is better):

| Predictor | All walking | During turns |
|---|---|---|
| Constant velocity (baseline) | 0.226 | 0.273 |
| Learned, positions only | 0.183 | 0.239 |
| **Learned + body/head direction** | **0.135** | **0.175** |

**Not switching to the wrong person (E3).** Wrong-person switches per 60 s episode:

| Target memory | All scenarios | With a look-alike nearby |
|---|---|---|
| Follow the tracker's ID only | 2.33 | 2.26 |
| **Gated appearance memory (ours)** | **0.10** | **0.02** |
| Ungated memory (always update) | 0.55 | 1.56 ← drifts to the look-alike |

Searching where the target was heading recovers the target in **75%** of loss events, vs **40%** for
stop-and-scan.

**Being robust to bad perception (E5, random mix).** Fraction of time the target stays in view at 1–3 m:

| Perception noise | Simple follower | Full system |
|---|---|---|
| half | 0.79 | 0.77 |
| normal | 0.74 | 0.78 |
| double | 0.62 | **0.74** |

**Honest negatives.** Steering toward the *predicted* position helps only a little overall (+1 to +2 points of
tracking; clearest in the crowded corridor: +8 points, turn lag −3 s). The best predictor does not follow better
than a simple one, probably because the target is slow (~0.35 m/s), so a 0.5 s look-ahead moves the goal only
~20 cm. Per-scenario tables with paired confidence intervals are in `results/E2/`.

> ⚠️ **Simulation only.** Human motion and the "look before you turn" cue are synthetic. These results show that the
> method can use such signals, not that real people behave exactly this way.

## How it is built: five stages

| Stage | What | Where it runs | State |
|---|---|---|---|
| **A** Cluster workflow | Sync code, build the Isaac Lab container, submit/fetch jobs | CHTC (HTCondor + Apptainer) | ✅ |
| **B** K1 walking policy | Our own RL walker (Isaac Lab, PPO, 4096 robots in parallel), Booster's official gains and walking pose | CHTC GPU | 🔄 training |
| **C** Following brain | Fast 2D simulator, perception noise, tracker, memory, predictor, planner, 8 test scenarios | Laptop CPU | ✅ |
| **D** Closed loop in Isaac Sim | Walking K1 + brain in 3D scenes with walking people; 3-view videos | CHTC GPU | 🔄 code ready |
| **E** Report | Tables, plots, videos | — | later |

### Test scenarios

| | Scenario | | Scenario |
|---|---|---|---|
| T1 | Sharp corner turn | T5 | Target disappears behind an obstacle |
| T2 | Exits a room and turns | T6 | Crowded corridor (6–10 people) |
| T3 | Someone walks between robot and target | T7 | Random mix (0–8 people) |
| T4 | Look-alike walks across the target's path | T8 | Dense open hall (10–15 people) |

Layouts come from a procedural generator (offices, open halls, corridor loops). Real building interiors come from
[InteriorGS / SAGE-3D](https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz): a shopping mall,
a hotel lobby and a museum, rendered as 3D Gaussian splats in Isaac Sim.

## Tech details

- **2D simulator (`pf/world`)**: 10 Hz, 0.05 m occupancy grids, A*-planned purposeful walkers with social-force
  avoidance, a synthetic head-turn cue 0.3–0.8 s before turns, view-dependent appearance embeddings.
- **Perception (`pf/perception`)**: field of view and occlusion (walls and people), distance-dependent detection and
  noise, false positives, constant-velocity Kalman tracker with Hungarian association and injected ID swaps.
  One knob (`noise_scale`) scales everything; 0 = perfect perception.
- **Memory (`pf/memory`)**: score = 0.6 × appearance (top-3 cosine to memory) + 0.4 × motion likelihood. The memory
  grows only when the match is confident, the person is isolated (no one within 1 m), and the view is new.
- **Predictor (`pf/prediction`)**: GRU (128) over 1 s of noisy history → 3 possible 2 s futures + probabilities;
  trained on ~250k samples; rotation-invariant inputs.
- **Planner (`pf/control`)**: Dynamic Window Approach over the robot's own depth-ray map (never the true map),
  with personal-space costs; controllers only choose the goal (1.5 m behind the target).
- **Walker (`k1_walker`)**: Isaac Lab 2.3.2 / Isaac Sim 5.1, 12 leg joints, 50 Hz policy, 200 Hz physics,
  gait clock + 5-step history + privileged critic; commands shaped like the brain's (ramps, sudden changes,
  stop-and-go, turn-in-place); domain randomization of friction, mass, gains, motor strength, delay and pushes.
- **Isaac closed loop (`isaac_follow`)**: scene built from the same 2D map (or an InteriorGS splat + collision
  mesh), people as kinematic mannequins, the K1 head camera drives perception, videos from chase/head/top views.

## Repository map

```
pf/              following brain + 2D simulator (world, perception, memory, prediction, control, eval)
k1_walker/       Isaac Lab K1 velocity task, train/play/eval scripts (Booster actuator models vendored)
isaac_follow/    Part D: Isaac scene builder, closed-loop runner, video composer
cluster/         CHTC scripts (sync, container build, jobs) + README_CLUSTER.md
configs/         world defaults + one YAML per experiment
models/          trained predictors (walker export lands here)
results/         CSVs, figures, GIFs per milestone/experiment
tests/           unit tests (world, perception, memory)
```

## Quick start (2D brain, laptop)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q                         # 33 tests
.venv/bin/python -m pf.world.demo                     # GIFs of the three layout families -> results/M3
.venv/bin/python -m pf.run_all --stage C --only E2    # an experiment -> results/E2
```

Cluster (walker training, Isaac videos): see [`cluster/README_CLUSTER.md`](cluster/README_CLUSTER.md).

## Credits and licenses

- Robot model and gains: Booster Robotics, [`booster_assets`](https://github.com/BoosterRobotics/booster_assets)
  (BSD-3), [`booster_train`](https://github.com/BoosterRobotics/booster_train) and
  [`booster_deploy`](https://github.com/BoosterRobotics/booster_deploy) (Apache-2.0; see `k1_walker/NOTICE`).
- Simulation: NVIDIA Isaac Sim / [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3), RSL-RL.
- Indoor scenes: [SAGE-3D / InteriorGS](https://github.com/Galery23/SAGE-3D_Official) (CC-BY-NC-4.0, research only).
- Compute: [UW–Madison CHTC](https://chtc.cs.wisc.edu/).

Pinned versions and attribution for vendored code: [`k1_walker/NOTICE`](k1_walker/NOTICE).
