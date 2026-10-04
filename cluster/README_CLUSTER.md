# Running Parts B and D on CHTC

All cluster values live in `cluster/cluster.env` (gitignored). Scripts print what they do, fail loudly, and are
safe to rerun.

## 0. Open the shared SSH connection (once per ~12 h)

CHTC login needs a password + Duo, so open one shared connection in **your own terminal**; every script reuses it:

```bash
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm/%r@%h:%p -o ControlPersist=12h -fN smittal39@ap2002.chtc.wisc.edu
```

## 1. Layout on CHTC

| Path | What |
|---|---|
| `/home/smittal39/k1-follow` | code (synced by `sync_up.sh`), job logs in `cluster/jobs/out/` |
| `/staging/s/smittal39/k1-follow/containers/isaaclab-2.3.2-k1.sif` | Isaac Lab 2.3.2 / Isaac Sim 5.1 + booster_assets (7.8 GB) |
| `/staging/s/smittal39/k1-follow/checkpoints/` | walker training logs/checkpoints (`walker_logs_<cluster>.tar.gz`) |
| `/staging/s/smittal39/k1-follow/videos/` | rendered videos |

## 2. Container (only if it must be rebuilt)

```bash
cluster/sync_up.sh
ssh ... 'cd ~/k1-follow/cluster && condor_submit build_container.sub'     # ~10 min on a build node
```
`build_container.sh` pulls `docker://nvcr.io/nvidia/isaac-lab:2.3.2` into a sandbox, adds booster_assets (pinned),
replaces Kit's cache/data/logs dirs with links into `/tmp/kit`, and packs a .sif. It avoids definition-file
builds, because those need fakeroot, which fails on CHTC build nodes. **If you rebuild, give the .sif a new
name**: OSDF caches files by name.

## 3. M0 check (hello world)

```bash
cluster/sync_up.sh
ssh ... 'cd ~/k1-follow/cluster && condor_submit hello_light.sub'   # any GPU with capability >= 7.5
ssh ... 'cd ~/k1-follow/cluster && condor_submit hello.sub'         # render-capable GPU (L40S/L40/A40/RTX PRO 6000)
cluster/fetch_results.sh                                          # -> results/cluster/jobs/hello_<id>.tar.gz
```

## 4. Part B: K1 walker

```bash
cluster/pack_code.sh                                              # sync + jobs/code.tar.gz
ssh ... 'cd ~/k1-follow/cluster && condor_submit submit_k1_stand.sub'        # M1 stand test
ssh ... 'cd ~/k1-follow/cluster && condor_submit submit_train_walker.sub'    # M2 training (self-checkpointing)
ssh ... 'condor_tail -f <cluster id>'                             # live training log
cluster/fetch_results.sh                                          # logs + exported policy
```
Training trains 500 iterations per chunk, then exits with code 85. HTCondor saves `./logs` and restarts the
job, which resumes from the newest checkpoint. At 4000 iterations it exports `k1_walker.pt` (TorchScript) and
`.onnx`. Override: `condor_submit TOTAL=5000 CHUNK=500 ENVS=4096 submit_train_walker.sub`.

## 5. Part D: Isaac closed loop

(to be added in M11)

## Notes

- GPU Lab per-user limits apply (short jobs ≤ 12 h, medium ≤ 24 h, long ≤ 7 d with at most 4 GPUs).
  Render-capable GPUs are often busy, so jobs that don't need rendering accept any GPU with compute
  capability ≥ 7.5 and ≥ 10 GB memory.
- Compute nodes have internet access (checked by the hello job), so Isaac assets can be fetched at runtime.
- The Isaac Lab stdout warnings "Extensions config 'extension.toml' doesn't exist ..." are harmless: Kit scans
  the job's scratch directory.
