# K1 person following: summary for the team

## What we built

A **Booster K1 humanoid in simulation that walks on its own learned legs and follows one chosen person through
crowded indoor spaces**: it keeps her in view at ~1.5–2 m, weaves around people and furniture, does not switch to a
look-alike, and searches when she disappears.

- **Walking:** our own RL walking policy for the K1 (Isaac Lab, PPO, 4,096 parallel robots, Booster's official
  gains and walking pose), trained on UW–Madison CHTC.
- **Following brain:** turns what the camera sees into *numbers about people*, remembers what the target looks like,
  predicts where she is going, and plans safe steps. Developed and tested in a fast 2D simulator.
- **Showcase:** the two combined in Isaac Sim, in NVIDIA's photorealistic warehouse and hospital, with realistic
  animated people (walk cycles retargeted onto the characters).

## Best videos

Each video shows three synchronized views: chase camera, the K1's own head camera (target boxed in red), and a top-down
map with the predicted path. Each scene is shown for the **simple follower (C0)** and the **full system**.

| Scene | Video (full system) | Compare (simple follower) |
|---|---|---|
| Warehouse aisles, crowded (T6) | `videos/photoreal/showcase_warehouse_T6_1000_full.mp4` | `..._C0.mp4` |
| Warehouse floor, 15-person crowd (T8) | `videos/photoreal/showcase_warehouse_T8_1000_full.mp4` | `..._C0.mp4` |
| Hospital, look-alike nearby (T4) | `videos/photoreal/showcase_hospital_T4_1000_full.mp4` | `..._C0.mp4` |
| Hospital, target disappears (T5) | `videos/photoreal/showcase_hospital_T5_1000_full.mp4` (rendering) | `..._C0.mp4` |

## Key numbers

1. **The walking K1 completed every photoreal run (6/6, 75 s each) without falling or touching anyone** in the full
   system; target kept in view at 1–3 m **98%** (warehouse aisles) and **97%** (dense crowd) of the time vs.
   85% / 92% for the simple follower.
2. **Zero wrong-person switches** in all 16 Isaac runs. In 2D tests on held-out layouts, our appearance memory cut
   wrong-person switches from **2.3 to 0.1 per episode** vs. following the tracker's ID; with a look-alike, 0.02.
3. **Searching where the person was heading** recovered a lost target **75%** of the time vs. **40%** for
   stop-and-scan.
4. **Walker:** tracks speed commands closely (0.4 m/s commanded → 0.41 m/s, 0.16 s delay; 0.8 rad/s turn → 0.82;
   stops in 0.3 s).
5. **Predicting the target's path** with body/head cues cut 2 s prediction error from 0.23 m to 0.14 m (0.27 → 0.18 m
   during turns).

## What worked

- Our own K1 walker, trained from Booster's official robot model, walking stably while being steered by the brain.
- "People as numbers": the brain never sees pixels, which made it fast to develop and easy to move into Isaac Sim.
- Gated appearance memory and predictive search: large, consistent gains.
- Photoreal scenes + animated people, sliced into maps the brain uses unchanged.

## What did not (yet)

- **Steering toward the predicted position helps only a little** (+1–2 points of tracking overall; clearer in crowded
  corridors). The target walks slowly, so looking 0.5 s ahead moves the goal only ~20 cm.
- **InteriorGS Gaussian-splat scenes could not be rendered** in our Isaac Lab container (NVIDIA's NuRec tools are
  missing), so we switched to NVIDIA's photoreal environments.
- Simulation only: human motion and the "look before turning" cue are synthetic; people do not react to the robot.
- Engineering bumps along the way (all fixed): arms held in a T-pose during the first training run, an exported
  policy fed the wrong initial pose, Isaac Sim hangs at start-up/shutdown on some cluster nodes, and a ~2-day CHTC GPU
  pause worked around via the national OSPool.

## Recommended next step

Re-run the quantitative comparison (Stage C experiments and a 10–20 episode Isaac check) with the **measured walker
response** and the denser crowds. Then, if the goal is the real robot, swap the simulated perception for a real
detector + tracker on the K1's camera (the brain's interface stays the same).
