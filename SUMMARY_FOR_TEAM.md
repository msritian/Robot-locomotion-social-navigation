# K1 person following: summary for the team

## What we built

A **Booster K1 humanoid in simulation that walks on its own learned legs and follows one chosen person through
crowded indoor spaces**: it keeps her in view at ~1.5–2 m, weaves around people and furniture, does not switch to a
look-alike, and searches when she disappears.

- **Walking:** our own RL walking policy for the K1 (Isaac Lab, PPO, 4,096 parallel robots, Booster's official
  gains and walking pose), trained on UW–Madison CHTC.
- **Following brain:** turns what the camera sees into *numbers about people*, remembers what the target looks like,
  predicts where she is going, and plans safe steps. Developed and tested in a fast 2D simulator.
- **Showcase:** the two combined in Isaac Sim, in NVIDIA's photorealistic warehouse, hospital and office, with realistic
  animated people (walk cycles retargeted onto the characters).

## Best videos

Each video shows three synchronized views: chase camera, the K1's own head camera (target boxed in red), and a top-down
map with the predicted path. Each scene is shown for the **simple follower (C0)** and the **full system**.

| Scene | Simple follower (C0): in view | Full system: in view | Videos |
|---|---|---|---|
| Office, crowd (T6) | 88% | **99%** | `videos/photoreal/showcase_office_T6_1000_{C0,full}.mp4` (in repo) |
| Hospital, look-alike nearby (T4) | 84% | **96%** | `videos/photoreal/showcase_hospital_T4_1000_{C0,full}.mp4` (in repo) |
| Office, target disappears (T5) | 100% | 100% | `showcase_office_T5_1000_*` |
| Hospital, target disappears (T5) | **85%** | 58% | `showcase_hospital_T5_1000_*` |
| Warehouse, dense crowd (T8) | **92%** | 86% | `showcase_warehouse_T8_1000_*` |
| Warehouse aisles (T6) | rendering | 87% | `showcase_warehouse_T6_1000_*` |

2-minute runs, one run per cell (demos, not statistics). In every run: **no falls, no wall/furniture hits, no
wrong-person switches**; 0–3 light bumps with people. Only the two pairs marked "in repo" are committed (size); the
rest are on the shared drive / cluster.

## Key numbers (2D simulator, ~50 random setups per scenario, layouts never seen in training)

1. **Wrong-person switches: 2.3 → 0.1 per episode** with the gated appearance memory (vs trusting the tracker's ID).
2. **Lost target found again: 40% → 75%** by searching where she was heading (vs stop-and-scan).
3. **Path prediction error at 2 s: 39 cm → 14 cm** (constant velocity → GRU with body/head direction; 19 cm without
   the body/head cues). During turns: 52 → 22 cm.
4. **Robust to bad detections:** with 2× detection noise the full system keeps the target in view 74% vs 62%.
5. **Overall following:** 76% → 77–78% in view (small gain); zero obstacle collisions.
6. **Walker:** 0.4 m/s commanded → 0.41 m/s (0.16 s delay); 0.8 rad/s → 0.82 rad/s.

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
