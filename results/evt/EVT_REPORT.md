# EVT-Bench evaluation: our follower vs OmTrackVLA

Status as of 2026-10-22. Pilots (§6–6b), full MP3D evaluation (§6c) and a held-out development-set check (§6d) are complete.

## 1. Goal

Compare our person-following system with OmTrackVLA, an open vision-language-action model, on **EVT-Bench**, the benchmark from the TrackVLA paper. Every system runs on the same episodes and is scored with the same metrics.

## 2. Benchmark

| Item | Value |
|---|---|
| Simulator | Habitat 3.0 (habitat-sim 0.3.1), with OmTrackVLA's habitat-lab fork and `evt_bench` (commit `e9cb1fb`) |
| Tasks | STT: single target. DT: distracted tracking, with look-alike people. AT: ambiguity tracking |
| Robot | Spot base on the navigation mesh. Action `[vx, vy, wz]` in [-1, 1] |
| Measured speeds per step | forward 0.357 m, sideways 0.217 m, turn 0.157 rad. A step is 0.25 s, so about 1.4 m/s, 0.9 m/s and 0.63 rad/s |
| Camera | Jaw RGB plus panoptic labels, 384×384, 90° field of view. Our agent also uses a jaw depth camera (see §5) |
| Episode | Up to 300 steps. Ends early if the robot is more than 4 m from the target for more than 20 steps ("Lost"), or on a collision |
| Humans | Target and distractors walk up to 0.25 m per step (we measured 0.05–0.16 m) |

**Metrics** (identical to TrackVLA):
- **SR, success rate:** at the end of the episode the robot is 1–3 m from the target and facing it.
- **TR, tracking rate:** fraction of steps with the target within 3 m and in view, meaning 3,000 px up to 30% of the image in the panoptic labels.
- **CR, collision rate:** episodes in which the robot comes within 0.5 m of the target.

## 3. Data used

- **Scenes:** only the **Matterport3D (MP3D)** half of EVT-Bench, because we have MP3D access. These are the 14 scenes: E9uDoFAP3SH, 2n8kARJN3HM, ac26ZMwG7aT, gYvKGZ5eRqb, VLzqgDo317F, Vt2qJdWjCF2, V2XKFyX4ASd, pRbA3pwrgk9, oLBMNvg9in8, YFuZgdQ5vWj, e9zR4mvMWw7, XcA2TqTSSAj, JmbYfDe2QKZ, HxpKQynjfin.
  - We pulled only the files for these 14 scenes out of the 16 GB `mp3d_habitat.zip`, using HTTP range requests (`evt/fetch_mp3d_scenes.py`), instead of downloading all of Matterport (about 130 GB).
- **Episodes:** **721 MP3D episodes per task**, out of 1,405 in total; the other half use HM3D scenes.
- **Tuning subset, used for all pilots below:** every 10th MP3D STT episode in file order, which gives **73 episodes from 10 of the 14 scenes**. We will **exclude these episodes from the final reported numbers**.
- **Humanoid models:** EVT-Bench's `humanoid_data`, downloaded from Google Drive.
- **OmTrackVLA weights:** `omlab/OmTrackVLA-0.6B`, plus SigLIP `google/siglip-so400m-patch14-384`, `Qwen/Qwen3-0.6B` and the gated DINOv3 `facebook/dinov3-vits16-pretrain-lvd1689m`.
  - All of them load from local copies with no network access.
  - The HF config's `llm_name` pointed to the authors' own machine, so we redirected it to our local Qwen3 copy.
- **Storage:** CHTC `/staging`, as single tar files because of the 1,000-file limit:
  - `evt_data.tar` (4.2 GB);
  - `hf_models.tar` (6.3 GB);
  - `dinov3_model.tar` (86 MB);
  - the container image `evt-habitat.sif` (4.3 GB).

## 4. Systems compared

**OmTrackVLA:** run unmodified through its own evaluation loop (`trained_agent.evaluate_agent`). Its inputs are the RGB image and the episode's language instruction.
- We added a strict check before any episode runs, because its code swallows loading errors: if a model is missing, it would silently act on garbage.
- Note: OmTrackVLA was trained on EVT-Bench training data, so it is evaluated **in distribution**.

**Ours:** the modular follower (`pf/`), adapted for Habitat in `evt/ours_agent.py`. The evaluation loop is a copy of OmTrackVLA's, with the same termination rules and metrics.

| Part | How it works |
|---|---|
| Person detection | Panoptic person masks with no identity, plus jaw depth, giving the position. Visibility check from §5 |
| Appearance | 256-D HSV colour histogram of the upper and lower body |
| Tracking | Kalman tracker with gating; 2 hits to confirm a track |
| Target memory | M2, gated continual appearance memory |
| Motion predictor | P0, constant velocity. Our learned predictor P2 expects 10 Hz history and EVT runs at 4 Hz |
| Follow controller | C1, keeping 1.3 m behind the target |
| Local planner | DWA (sampling planner) over a local obstacle map built from 32 depth rays |
| Search when lost | S1, predictive search |
| Target selection | **First-frame protocol:** the person in view at the start is the target. OmTrackVLA gets the language instruction instead |

## 5. Problems found and fixed, in order

| # | Problem | Symptom | Fix |
|---|---|---|---|
| 1 | The benchmark's main-human sensor keeps the **first** episode's semantic ID; the rendered target uses that ID, not the ID in each episode's info | Target never detected | Read the ID from the benchmark's own sensor. This quirk affects every agent the same way |
| 2 | Robot drove at a target that was walking toward it | Collisions | `hold_off_when_approached`, follow distance 1.3 m |
| 3 | The stock jaw **depth** camera on Spot is mounted on the **arm** (link 6, rotated); RGB and panoptic sit on the body | Positions wrong by up to several metres | Re-mount depth to match the panoptic camera every episode. Probe error is now 0.06–0.09 m |
| 4 | The camera sees the **robot's own body** in the lower middle of the view, at 0.18 m | 16 of 32 obstacle rays always at 0.18 m, so the planner believed every move hit a wall and froze; 40% collisions | Ignore depth closer than 0.35 m |
| 5 | The panoptic render shows people **through walls** | Target measured at the wall's distance (1–3 m too close) | Visibility check: keep the farthest depth cluster of the mask; on each side the person must stand clearly in front of the background, or next to an occluder edge |
| 6 | Target walks straight at the robot; no safe move exists, so the planner stopped in its path | Collisions | Option `give_way`: step to the side of the person's path. Option `evade_fallback`: take the wall-safe move with the most distance |
| 7 | After losing the target, the robot gets stuck behind walls (straight-line goal cost) and rejects almost every move (1.5 s constant-command rollouts) | Most "Lost" episodes | Option `path_cost`: walking distance on the robot's own map. Option `brake_after_s` = 0.5 (hold the move 0.5 s, then brake), for walls only (`brake_walls_only`). Fewer Lost, but more collisions with the target (see §6) |

New behaviour is switched on by config options, with defaults off, so earlier non-EVT results stay reproducible. All 33 unit tests pass.

## 6. Results on the 73-episode tuning subset (STT, MP3D)

All rows use the same 73 episodes.

| System / version | SR % | TR % | CR % | Lost / Normal / Collision |
|---|---|---|---|---|
| Ours, first run (fixes 1–2) | 4.1 | 33.7 | 9.6 | 57 / 9 / 7 |
| Ours + depth re-mounted (fix 3) | 0.0 | 30.8 | 21.9 | 53 / 4 / 16 |
| Ours + self-body and visibility fixes (4–5) | 28.8 | 55.4 | 2.7 | 39 / 32 / 2 |
| Ours + give way and evade (6) | **37.0** | 56.9 | 5.5 | 34 / 35 / 4 |
| Ours + (6) with **oracle target identity** | 37.0 | 56.9 | 5.5 | 34 / 35 / 4 |
| Ours + (6) + brake for walls only (`evgw_bw`) | 32.9 | 64.5 | 17.8 | 25 / 35 / 13 |
| Ours + (6) + path cost + brake for walls (`evgw_pbw`) | 37.0 | 64.8 | 20.5 | 25 / 33 / 15 |
| **OmTrackVLA** | **87.7** | **87.3** | **0.0** | 5 / 68 / 0 |

For reference, the TrackVLA paper reports about 85% SR on STT using all scenes.

Diagnostics:
- **Recognition is not our bottleneck on STT.** The oracle run, where the target's identity is given, is identical to the normal run. Other people are in view only 4% of the time, and we never switched to the wrong person.
- **Detection works.** When the benchmark counts the target as visible, we detect it 97% of the time. The median position error is 0.05 m.
- **Main failure: losing the target after it leaves the view.** This happens mostly around corners and doorways, where search and the planner got stuck (problem 7).
- On the first 10 episodes, OmTrackVLA got 90% SR versus our 0% before the fixes.

### DT and AT (same 73-episode subset positions, MP3D)

| Task | System | SR % | TR % | CR % | Lost / Normal / Collision |
|---|---|---|---|---|---|
| DT | Ours (`evade_gw`) | 19.2 | 50.9 | 11.0 | 39 / 26 / 8 |
| DT | OmTrackVLA | 38.4 | 62.4 | 4.1 | 29 / 41 / 3 |
| AT | Ours (`evade_gw`) | 8.2 | 45.4 | 16.4 | 47 / 14 / 12 |
| AT | OmTrackVLA | 52.1 | 76.2 | 5.5 | 22 / 47 / 4 |

OmTrackVLA drops from 88% to 38% once distractors appear. In DT, when the VLA is clearly heading for someone, it is the **wrong person 37% of the time**. That is the gap the hybrid targets.

## 6b. Hybrid (direction C), development pilots

`evt/hybrid_agent.py`: OmTrackVLA drives. Our side runs on the same frames: person masks without identity plus depth, tracker, appearance memory. Our memory learns the target from the person the VLA heads for over its first 6 steps, so it gets no extra target information.

| Run | What is on | STT SR / TR / CR | DT SR / TR / CR |
|---|---|---|---|
| h1 | DWA shield (walls + people, 1.5 s) + takeover on distractor and on search | 17.8 / 49.0 / 0.0 | 11.0 / 42.0 / 0.0 |
| h0 | nothing (sanity check: must equal the VLA alone) | 87.7 / 87.7 / 0.0 | 34.2 / 61.8 / 5.5 |
| hS | DWA shield, people only | 87.7 / 87.1 / 0.0 | 24.7 / 55.5 / 2.7 |
| hT | takeover (distractor + search) | 75.3 / 83.2 / 0.0 | 39.7 / 61.7 / 5.5 |
| h2 | hS + hT | 72.6 / 81.8 / 0.0 | 34.2 / 57.7 / 5.5 |

What we learned:
- **h0 matches the VLA alone**, so the wiring is correct. The 34.2 vs 38.4 on DT is run-to-run variation, about 3 episodes.
- **Wall shield (h1): harmful on this benchmark.** It overrode the VLA on 42% of steps. The cause is real 0.5 m corridor walls, not sensor error. But Habitat's base slides along walls on the navigation mesh, and only people end an episode, so wall checks can only slow the VLA down. They stay available (`--shield-walls`) for the real robot.
- **People shield with a 1.5 s look-ahead (hS):** halves collisions on DT but loses the target more often. Replaced by a 0.5 s look-ahead that only slows the VLA's own command.
- **Search takeover: harmful.** Episodes with 30+ steps of our search turned VLA successes into Lost. The VLA re-finds the target better than our search does here.
- **Distractor takeover: helpful.** When it fires, our memory has the right person and the VLA the wrong one 90% of the time. It flipped 7 DT failures into successes.
- Our memory locked onto the wrong person at the start in 18 of 68 DT episodes, because the VLA headed for the wrong person in its first steps. This limits how much the takeover can fix.

### Round 2: distractor takeover only (no search takeover)

| Run | Setup | STT SR / TR / CR | DT SR / TR / CR | AT SR / TR / CR |
|---|---|---|---|---|
| OmTrackVLA alone | – | 87.7 / 87.3 / 0.0 | 38.4 / 62.4 / 4.1 | 52.1 / 76.2 / 5.5 |
| hD | strict takeover (agree 10 steps, memory conf. 0.85), no shield | 87.7 / 87.8 / 0.0 | 38.4 / 63.9 / 6.8 | 57.5 / 77.7 / 4.1 |
| hDL | loose takeover (agree 3, conf. 0.75), no shield | 87.7 / 87.4 / 0.0 | 39.7 / 66.1 / 6.8 | 57.5 / 77.8 / 4.1 |
| hDLs | hDL + 0.5 s people shield | 87.7 / 87.1 / 0.0 | 32.9 / 59.6 / 1.4 | 46.6 / 77.3 / 5.5 |
| hX | strict takeover + **text grounding** init (SigLIP) | 87.7 / 87.1 / 0.0 | 41.1 / 63.2 / 4.1 | 53.4 / 77.7 / 4.1 |

- The hybrid no longer hurts on STT. On DT and AT it gains 1–5 points of SR and up to 4 points of TR. **On 73 episodes that is within noise** (±3 episodes, about ±4 points), so the full run will decide.
- **The people shield cuts collisions with the target (DT CR 6.8 → 1.4) but costs success.** It stops the robot when distractors walk past. A target-only version is being tested.
- **Text grounding does not work here.** In frames with several people, SigLIP ranks the true target first only 42% (DT) and 27% (AT) of the time, about chance. The VLA's heading is right 68–76% of the time. Distractors look alike, and person crops are small (384×384 frames). With text init, memory started on the wrong person in 31 of 72 DT episodes, versus 14 with VLA init. Not used further.
- **What still fails on DT (hDL, 31 failures):**
  - 15: target out of view at the end, with the robot near a distractor;
  - 7: VLA following a distractor;
  - 5: collisions with the target;
  - 4: target in view but not followed.

  The first group needs re-finding a target that left the view while distractors are around. Our search did this worse than the VLA on STT.

## 6c. Full evaluation (MP3D, 648 episodes per task, tuning subset excluded)

| Task | System | SR % | TR % | CR % | Normal / Lost / Collision |
|---|---|---|---|---|---|
| STT | OmTrackVLA | 76.9 | 81.5 | 3.9 | 548 / 75 / 25 |
| STT | Ours (`evade_gw`) | 26.1 | 55.6 | 12.0 | 256 / 314 / 78 |
| DT | OmTrackVLA | 39.8 | 60.8 | 10.0 | 351 / 232 / 65 |
| DT | Ours (`evade_gw`) | 19.9 | 50.7 | 12.7 | 218 / 348 / 82 |
| AT | OmTrackVLA | 54.0 | 74.0 | 7.7 | 424 / 174 / 50 |
| AT | Ours (`evade_gw`) | 8.3 | 49.9 | 19.4 | 130 / 392 / 126 |

**Hybrid vs OmTrackVLA, same 648 episodes per task:**

| Task | System | SR % | TR % | CR % | Episodes vs VLA: hybrid wins / losses | McNemar p |
|---|---|---|---|---|---|---|
| STT | OmTrackVLA | 76.9 | 81.5 | 3.9 | | |
| STT | **Hybrid hDLt** (distractor takeover + target-only 0.5 s shield) | **77.2** | 81.8 | 4.0 | 13 / 11 | 0.84 |
| STT | Hybrid hDL (distractor takeover only) | 75.6 | 81.3 | 4.0 | 6 / 14 | 0.12 |
| DT | OmTrackVLA | 39.8 | 60.8 | 10.0 | | |
| DT | **Hybrid hDLt** | **42.3** | 61.4 | 9.4 | 55 / 39 | 0.12 |
| DT | Hybrid hDL | 41.5 | 61.7 | 11.0 | 56 / 45 | 0.32 |
| AT | OmTrackVLA | 54.0 | 74.0 | 7.7 | | |
| AT | Hybrid hDLt | 54.5 | 73.7 | 7.9 | 40 / 37 | 0.82 |
| AT | **Hybrid hDL** | **56.9** | 74.1 | **6.8** | 50 / 31 | **0.045** |

Both hybrid configurations were fixed on the tuning subset before this run. hDLt was named the main one in advance; hDL is reported as the ablation.

- **The hybrid is never worse than the VLA alone** (beyond noise) and is ahead on DT and AT. Only hDL on AT is significant at 5% (+2.9 SR, p = 0.045). No correction was made for testing 6 comparisons, so treat that as weak evidence.
- **The shield barely changes collisions** (DT 10.0 → 9.4). In the full DT run, 44 of 61 collisions happened while our memory was still learning the target. That is mostly in the first ~18 steps, with the target walking into a stopped robot, where a target-only shield has nobody to protect against.
  - The tuning subset had too few collisions (3–6 per run) to show this. We found it on the evaluation set, so a fix must be developed on other episodes, then evaluated once more.
- 2 of the 72 STT chunks first ran on a machine without a working GPU driver (`gitter2003`, EGL error). They were rerun, and the machine is now excluded in `submit_evt.sub`.

On the full set, OmTrackVLA's STT SR is lower than on the tuning subset (76.9 vs 87.7), and its collision rate is higher on every task. The tuning subset happened to be easier.

## 6d. Held-out development set (MP3D train split, 151 episodes per task)

To try new ideas without tuning on the evaluation episodes, a separate development set was built from 8 MP3D scenes of EVT-Bench's **train** split: `ur6pFq6Qu1A PuKPg4mmafe jtcxE69GiFV VFuaQ6m2Qom Z6MFQCViBuw mJXqzFtmKg4 5q7pvUzZiYa QUCTc6BB5sX`. None of these scenes are in the 14 val scenes. Every 3rd episode was used, giving 151 per task. Two new shields were tested:
- **hE (evade):** if any person's 2-step prediction comes within 0.5 m, scale the VLA command down; if even stopping is unsafe, the DWA steps around people.
- **hES:** hE plus a narrow search takeover. It acts only when the target has been unseen for more than 2 s and the VLA is not heading to anyone.

Results (SR / TR / CR, %):

| System | STT | DT | AT |
|---|---|---|---|
| OmTrackVLA | **78.1** / 81.6 / 6.0 | **31.8** / 58.2 / 13.9 | **56.3** / 74.3 / 11.3 |
| Hybrid hDLt | 75.5 / 81.6 / 7.3 | 31.1 / 59.3 / 18.5 | 53.0 / 73.7 / 12.6 |
| Hybrid hE | 77.5 / 81.6 / 6.6 | 29.1 / 57.2 / 16.6 | 52.3 / 73.8 / 9.3 |
| Hybrid hES | 77.5 / 81.8 / 6.6 | 29.1 / 56.7 / 14.6 | 47.7 / 71.4 / 9.3 |

- **On new scenes, no hybrid beats OmTrackVLA alone.** All differences are within 1–5 episodes, except hES on AT (−8.6 SR), where the search takeover clearly hurts.
- **The val-set gains of hDLt (DT +2.5, §6c) do not carry over** (DT −0.7 here). They were most likely noise, as the McNemar p-values in §6c already suggested.
- hDLt collides *more* than the VLA on DT (28 vs 21 episodes). Distractor takeover sometimes drives our memory's target into people.
- Evade lowers AT collisions (17 → 14 episodes) but loses about as many episodes to "Lost", so success does not improve.

**Conclusion:** with OmTrackVLA driving, our memory, takeover and shield layers do not make following more accurate. The VLA already keeps the target in view better than our perception, and most remaining failures are recognition errors (following a distractor, or losing the target among people) that our layers cannot fix without a better recogniser. No new configuration earned a second evaluation-set run. The §6c numbers stay the final ones, reported as "hybrid ≈ OmTrackVLA".

## 7. Caveats and fairness notes

- §6–6b are **pilot numbers on a 73-episode tuning subset**. Final numbers (§6c) use the other 648 MP3D episodes per task; §6d uses separate train-split scenes.
- Only the MP3D half of EVT-Bench (no HM3D), so numbers are not directly comparable to the paper's all-scene averages.
- **Inputs differ.** Our agent uses an extra depth sensor and the panoptic masks (with identity hidden) as its person detector. OmTrackVLA uses RGB plus language only.
- **Target specification differs.** Ours uses the first frame; OmTrackVLA uses the language instruction. This matters most on DT and AT.
- **Training distribution.** OmTrackVLA is trained on EVT-Bench data; our system was designed for the K1 in a different simulator.
- We use the P0 predictor instead of P2 because of the 4 Hz step rate.
- The benchmark's own visibility metric also "sees" through walls (fix 5 only affects our detector).
- In the oracle run, other people are not detected at all, so they are not avoided either. That slightly favours the oracle run.

## 8. Decision and next steps

**Chosen direction: C, a hybrid.** OmTrackVLA-style recognition and keeping the target in view, combined with our structured memory, search and re-finding after occlusion, and our safety planner and controller.

1. Done: problem 7 pilots, DT and AT pilots, the first hybrid and its ablations (§6b).
2. Tune the hybrid: round 3 (target-only shield) running.
3. Full runs **launched 2026-10-19** for OmTrackVLA and ours (`evade_gw`): all MP3D episodes minus the tuning subset (`--exclude-every 10`), 648 per task, 72 jobs per task. The hybrid's full run follows once its configuration is fixed.
4. Done (§6c, §6d): full evaluation of the hybrid, then a held-out development set check of new shields. Neither beats OmTrackVLA reliably.
5. TrackVLA and TrackVLA++: no public weights found (2026-10-19). Candidate stronger public baseline: ReferTrack-Qwen3-4B (`hjyeee/ReferTrack-Qwen3-4B`), whose paper reports AT SR 74.1. Not yet checked.

## 9. Reproducing

- Code:
  - `evt/`: `run_evt.py`, `ours_agent.py`, `evt_cfg.py` (variants), `summarize.py`, `replay.py`, `fetch_mp3d_scenes.py`;
  - `cluster/`: `build_habitat.*`, `evt_eval.sh`, `submit_evt.sub`, `wait_q.sh`, `get_dinov3.sh`.
- Pilot run:
  ```
  condor_submit AGENT=ours TASK=stt N=8 EVERY=10 VAR=<variant> [ORACLE=1] submit_evt.sub
  ```
- OmTrackVLA: the same command with `AGENT=omtrackvla` and `MODELS=", osdf:///…/hf_models.tar, osdf:///…/dinov3_model.tar"`.
- Summary: `python3 evt/summarize.py results/evt/<run>/out/<agent>/stt [...]`. With several folders it compares only the episodes they share.
- Local result folders under `results/evt/`:
  - `rec`: first run;
  - `fix1`: depth re-mounted;
  - `p_base`: self-body and visibility fixes;
  - `p_evgw`: give way and evade;
  - `p_orac`: oracle identity;
  - `omt1`, `omt_dt`, `omt_at`: OmTrackVLA;
  - `ours_dt`, `ours_at`, `p_bw`, `p_pbw`: ours;
  - `h0`, `h1`, `hS`, `hT`, `h2` `_stt`/`_dt`: hybrid ablations;
  - `diag`: images of the bad frames.
