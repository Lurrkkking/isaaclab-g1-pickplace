# G1 Locomanip Generated Demo Replay Debug Summary

## Background

Debug target:

- Replaying `generated_dataset_g1_locomanip_20.hdf5` caused G1 to flip / diverge badly.
- The goal was to locate the replay failure reason without re-installing packages and without focusing on video first.

Dataset under test:

- `/root/autodl-tmp/IsaacLab/datasets/generated_dataset_g1_locomanip_20.hdf5`

Known facts at the start:

- `actions_shape = (220, 32)`
- `processed_actions_shape = (220, 43)`
- Generation task: `Isaac-Locomanipulation-G1-Abs-Mimic-v0`
- Training / play task: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`

---

## Main Findings

### 1. `env.step()` should consume `actions`, not `processed_actions`

This was confirmed from the action system and the official replay path.

- `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` has env action dim `32`
- Upper body IK raw action dim = `28`
- Lower body locomotion raw action dim = `4`
- Total env action dim = `32`

Source evidence:

- `scripts/tools/replay_demos.py` uses `EpisodeData.get_next_action()`
- `EpisodeData.get_next_action()` reads only `actions`
- `processed_actions` are recorder-side post-processed internal action targets

Conclusion:

- `actions` are the correct replay input
- `processed_actions` must not be fed into `env.step()`

### 2. `initial_state` exists and restores correctly

The dataset contains:

- `actions`
- `processed_actions`
- `initial_state`
- `states`

`demo_0/initial_state` includes:

- robot root pose / root velocity
- robot joint position / joint velocity
- object root pose / root velocity

The first-step diagnostic showed:

- `reset_to_vs_initial_state diff ... max=0.000000`

Conclusion:

- Replay failure was not caused by missing `initial_state`

### 3. Replaying generated data in the normal env was wrong

The generated HDF5 stores:

- `env_args.env_name = Isaac-Locomanipulation-G1-Abs-Mimic-v0`

However, the failing replay attempts were using:

- `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`

This was a real mismatch.

Even after fixing action usage and restoring `initial_state`, replaying in the normal env already showed step-0 drift.

Conclusion:

- Generated G1 locomanip demos must be replayed in the dataset's Mimic env
- Replaying them in the normal env can produce divergence / flipping

### 4. The official `replay_demos.py` had multiple bugs for this workflow

The script was not ready for replaying this generated Mimic dataset out of the box.

Bugs found:

1. It did not import `isaaclab_mimic.envs` / `isaaclab_mimic.envs.pinocchio_envs`
   - Result: dataset env name existed in HDF5, but gym could not resolve it

2. It parsed `env_name` from the dataset, but still called:
   - `gym.make(args_cli.task, cfg=env_cfg)`
   - If `--task` was omitted, this became `gym.make(None, ...)`

3. Its state validator had a shape bug
   - dataset state tensors were `(1, D)`
   - runtime comparison tensors were `(D,)`
   - the validator raised before doing a meaningful comparison

4. Headless mode still tried to initialize keyboard teleoperation helpers
   - fixed to skip in headless runs

### 5. Even in the correct Mimic env, state-by-state replay is not exact

After fixing the script bugs and replaying in `Isaac-Locomanipulation-G1-Abs-Mimic-v0`, `--validate_states` still reported mismatches from action-index `0`.

The mismatch pattern was concentrated in:

- robot `root_velocity`
- robot `joint_position`
- robot `joint_velocity`

while object pose was much closer.

Conclusion:

- The remaining issue is not `actions` vs `processed_actions`
- It is not `initial_state`
- It is not just "wrong env id" anymore once replay is moved into Mimic
- There is still non-exact replay drift from step 0 under the current runtime conditions

Most likely causes at this stage:

- runtime / physics / device path mismatch
- generated data not being strictly state-replayable under current replay conditions

This explains why replay can visually stay reasonable when run on the corrected path, but not match the stored `states` exactly.

---

## HDF5 Structure Observed

For `demo_0`, the dataset contains:

- `actions (220, 32)`
- `processed_actions (220, 43)`
- `initial_state/...`
- `states/...`
- `obs/...`

At `/data` attrs:

- `env_args = {"env_name": "Isaac-Locomanipulation-G1-Abs-Mimic-v0", "type": 2, "sim_args": {...}}`

Important note:

- The HDF5 stores env name and sim args
- It does not store action normalization metadata in the replay path we inspected

---

## Code Changes Made

### 1. Fixed wrong `processed_actions` fallback

File:

- `scripts/g1_locomanip_replay_video.py`

Change:

- removed fallback that switched replay input from `actions` to `processed_actions`
- now replay only accepts raw `actions` with the correct env action dimension

### 2. Fixed `replay_demos.py` for Mimic datasets

File:

- `scripts/tools/replay_demos.py`

Changes:

- import Mimic env registrations when pinocchio is enabled
- use dataset `env_name` when creating the environment
- skip keyboard interface in headless mode
- print explicit replay action source and action dim
- fix state validation shape handling

### 3. Added one-step diagnostic script

File:

- `scripts/g1_locomanip_first_step_diag.py`

Purpose:

- restore `initial_state`
- apply exactly one recorded action
- print root pose / base height / joint pos
- compare runtime state vs stored dataset state

### 4. Added a new MP4 replay script that avoids the missing viewport module

File:

- `scripts/g1_locomanip_replay_mp4.py`

Purpose:

- replay `demo_N` in the dataset's own Mimic env
- capture frames using `render_mode="rgb_array"`
- encode MP4

Reason this was needed:

- the previous viewport-based video scripts depended on `omni.kit.viewport.utility`
- that module was missing in the current environment

---

## Final Outcome

We now have a working path to generate a non-flipping `demo_0` replay MP4 from the corrected replay route.

Generated files:

- `/root/autodl-tmp/IsaacLab/videos/g1_locomanip_demo_0.mp4`
- `/root/autodl-tmp/IsaacLab/videos/g1_locomanip_demo_0_smoketest.mp4`

This path:

- uses raw `actions`
- restores `initial_state`
- replays in the dataset's Mimic env
- avoids the broken viewport dependency

---

## Recommended Replay Command

```bash
cd /root/autodl-tmp/IsaacLab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/isaacsim-5.1.0

unset LD_LIBRARY_PATH
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p scripts/g1_locomanip_replay_mp4.py \
  --device cpu \
  --dataset_file /root/autodl-tmp/IsaacLab/datasets/generated_dataset_g1_locomanip_20.hdf5 \
  --demo_key demo_0 \
  --fps 10 \
  --output /root/autodl-tmp/IsaacLab/videos/g1_locomanip_demo_0.mp4 \
  --headless \
  --enable_pinocchio
```

---

## Short Version

The replay failure was not caused by missing `initial_state`, and after the first pass it was no longer caused by `actions` vs `processed_actions` either.

The real failures were:

- replaying generated Mimic data in the wrong env
- multiple bugs in the official replay script
- a broken viewport-based video path in the current environment

After fixing those, we were able to export a stable `demo_0` MP4 using a new `rgb_array`-based replay script.
