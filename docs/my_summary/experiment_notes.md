# Experiment Notes: G1 Locomanipulation Imitation Learning

## 1. Objective

This note records the reproduction experiments for the Isaac Lab G1 locomanipulation pick-and-place imitation learning pipeline.

The goal is not to propose a new algorithm, but to verify the full official-style workflow:

```text
official annotated demonstrations
→ Mimic demonstration generation
→ robomimic BC-RNN training
→ closed-loop policy rollout evaluation
```

The reproduced rollout task is:

```text
Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

The Mimic generation task is:

```text
Isaac-Locomanipulation-G1-Abs-Mimic-v0
```

The policy is trained using low-dimensional observations and a robomimic BC-RNN policy.

---

## 2. Pipeline

The reproduction pipeline contains four stages.

### 2.1 Official annotated dataset

The official annotated G1 locomanipulation dataset is used as the seed demonstration source:

```text
dataset_annotated_g1_locomanip.hdf5
```

This file is not directly used as the final training dataset. It is used as the input to Isaac Lab Mimic.

### 2.2 Mimic demonstration generation

Isaac Lab Mimic is used to generate additional successful demonstrations.

Example command:

```bash
./isaaclab.sh -p -u scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
  --device cpu \
  --headless \
  --num_envs 4 \
  --generation_num_trials 1000 \
  --enable_pinocchio \
  --input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ./datasets/generated_dataset_g1_locomanip_1000.hdf5
```

The generated dataset contains successful demonstration trajectories with raw environment actions.

Important detail:

```text
actions: 32-D raw actions used by env.step()
processed_actions: 43-D internal expanded action targets
```

`processed_actions` should not be directly fed into `env.step()`.

### 2.3 BC-RNN training

The generated demonstrations are used to train a robomimic BC-RNN policy.

Example command:

```bash
./isaaclab.sh -p -u scripts/imitation_learning/robomimic/train.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --algo bc \
  --normalize_training_actions \
  --dataset ./datasets/generated_dataset_g1_locomanip_1000.hdf5 \
  --epochs 1800 \
  --name bc_rnn_low_dim_g1_1000demo
```

When `--normalize_training_actions` is used, the corresponding `normalization_params.txt` must be used during policy rollout. The checkpoint and normalization parameters must come from the same training run.

### 2.4 Policy rollout evaluation

The trained checkpoint is evaluated using closed-loop rollouts:

```bash
./isaaclab.sh -p -u scripts/imitation_learning/robomimic/play.py \
  --device cpu \
  --enable_pinocchio \
  --headless \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --num_rollouts 50 \
  --horizon 1000 \
  --checkpoint <checkpoint_path> \
  --norm_factor_min <run_specific_min> \
  --norm_factor_max <run_specific_max>
```

The reported metric is the number of successful rollouts out of the total number of trials.

---

## 3. Results

| Generated demos | Training epochs | Evaluation rollouts | Successful rollouts | Success rate |
|---:|---:|---:|---:|---:|
| 100 | 50 | 10 | ~1 | ~10% |
| 500 | 100 | 30 | 8 | ~27% |
| 1000 | 300 | 50 | 27 | 54% |
| 1000 | 500 | 50 | 31 | 62% |
| 1000 | 1800 | 50 | 48 | 96% |

The strongest reproduced result so far is:

```text
1000 Mimic-generated demonstrations
1800 training epochs
48 / 50 successful rollouts
96% preliminary success rate
```

This indicates that the official Mimic + robomimic BC-RNN pipeline can produce a high-success G1 locomanipulation policy after sufficient data generation and training.

---

## 4. Observations

### 4.1 Demonstration replay is not policy rollout

A generated demonstration replay only verifies that the stored demonstration actions can be replayed correctly. It does not prove that a trained policy has learned the task.

In this project, both stages were checked separately:

```text
generated demo replay
policy checkpoint rollout
```

The final result is based on closed-loop policy rollout, not just demonstration replay.

### 4.2 More demonstrations and longer training both matter

The success rate improved as the number of generated demonstrations and the number of training epochs increased.

The 500-demo policy already learned partial task behavior, but failures often happened during later placement or release. With 1000 generated demonstrations and longer training, the policy became much more stable.

### 4.3 Failure cases were stage-dependent

Early policies often showed partial success:

```text
approach object
→ grasp object
→ move object near the target
→ fail to release or place correctly
```

This suggests that the BC-RNN policy first learned the approach and grasp phases, while the release / placement phase required more data and longer training to become reliable.

### 4.4 Horizon affects evaluation but does not solve policy quality

Increasing the rollout horizon can help when the policy is slow, but it does not fix unstable behavior.

In early experiments, increasing `horizon` improved some trials. However, once the horizon was sufficiently long, remaining failures were mostly due to policy instability rather than time limit.

---

## 5. Key Debugging Points

Several implementation details were critical for successful reproduction.

### 5.1 Use raw actions for replay

The generated HDF5 dataset contains both:

```text
actions
processed_actions
```

For environment replay, the correct input is the raw 32-D `actions`.

Using `processed_actions` directly can cause severe motion errors, including robot flipping.

### 5.2 Restore initial state during replay

A generated demonstration is only valid under its recorded initial robot and object state.

A simple replay loop such as:

```python
env.reset()
for action in actions:
    env.step(action)
```

is not sufficient.

The replay must restore the demonstration `initial_state`.

### 5.3 Keep Mimic and normal environments separate

The Mimic generation environment and the normal training / rollout environment have different roles:

```text
Mimic generation:
Isaac-Locomanipulation-G1-Abs-Mimic-v0

Training and policy rollout:
Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

Mixing these incorrectly can lead to invalid replay or confusing results.

### 5.4 Match checkpoint with normalization parameters

When action normalization is enabled during training, the rollout command must use the `normalization_params.txt` from the same training run.

Using normalization values from a different run can severely degrade policy behavior.

### 5.5 Headless video export needs custom handling

The official replay / visualization tools are not always reliable in a pure headless server environment.

For this reproduction, custom frame-capture scripts were used to export generated demo replay and policy rollout videos.

---

## 6. Interpretation

This reproduction does not introduce a new learning algorithm. Its technical value is in verifying and analyzing the full humanoid locomanipulation imitation learning pipeline under a headless server setup.

The main reproduced components are:

```text
Isaac Sim / Isaac Lab headless execution
G1 locomanipulation task setup
Mimic demonstration generation
robomimic BC-RNN training
closed-loop policy rollout evaluation
debugging of action replay and normalization issues
```

The main result is that the reproduced BC-RNN policy reached 48/50 successful rollouts after training with 1000 generated demonstrations for 1800 epochs.

---

## 7. Current Limitations

This project is still a reproduction and analysis project, not a new algorithmic contribution.

Current limitations include:

- no real-robot deployment;
- no vision-language input;
- no comparison with VLA or diffusion-policy baselines;
- no new policy architecture;
- limited evaluation diversity.

The current policy is trained and evaluated in simulation under the Isaac Lab G1 locomanipulation task.

---

## 8. Possible Next Steps

Potential extensions include:

1. checkpoint sweep across different epochs;
2. failure mode statistics over more rollouts;
3. comparison between different dataset sizes under the same training schedule;
4. filtering low-quality generated demonstrations;
5. adding visual observation or language-conditioned high-level control;
6. using a VLA or VLM as a high-level planner while retaining the BC-RNN as a low-level continuous control policy.