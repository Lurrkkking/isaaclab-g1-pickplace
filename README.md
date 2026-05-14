# IsaacLab G1 PickPlace

<p align="center">
  <a href="#english"><b>English</b></a> |
  <a href="#中文"><b>中文</b></a>
</p>

<p align="center">
  This project reproduces the Isaac Lab G1 locomanipulation imitation learning pipeline and further extends it with a low-dimensional Diffusion Policy.<br>
  本项目复现 Isaac Lab G1 人形机器人移动操作模仿学习流程，并进一步接入 low-dimensional Diffusion Policy 完成闭环 rollout。
</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <b>robomimic BC-RNN Policy</b><br>
      <sub>1000 Mimic demos · 1800 epochs · 48/50 successful rollouts</sub><br><br>
      <img src="GIFs/rollout_1_1000demo_1800epoch.gif" width="420"/>
    </td>
    <td align="center" width="50%">
      <b>Low-dimensional Diffusion Policy</b><br>
      <sub>Same G1 demonstrations · diffusion action chunks · IsaacLab closed-loop rollout</sub><br><br>
      <img src="GIFs/diffusion_100.gif" width="420"/>
    </td>
  </tr>
</table>
---

## English

### Overview

This repository records my reproduction and debugging process for the **G1 pick-place / locomanipulation tasks in NVIDIA Isaac Lab**.

The work currently covers three parts:

1. Reproducing and debugging the fixed-base upper-body G1 pick-place environment:

```text
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
```

2. Reproducing the official Isaac Lab Mimic + robomimic imitation learning pipeline for G1 locomanipulation:

```text
Isaac-Locomanipulation-G1-Abs-Mimic-v0
Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

3. Extending the same G1 Mimic-generated demonstrations to a low-dimensional Diffusion Policy pipeline:

```text
G1 Mimic demonstrations
→ low-dimensional zarr dataset
→ Diffusion Policy training
→ IsaacLab closed-loop rollout
```

This project is based on the official NVIDIA Isaac Lab project:

```text
NVIDIA Isaac Lab
https://github.com/isaac-sim/IsaacLab
```

---

### Current Status

Verified:

- Isaac Sim 5.1 runs on a headless RTX 4090 server.
- Isaac Lab is installed from source.
- `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` can be registered, reset, and stepped.
- The fixed-base upper-body action dimension is verified as 28.
- PinkIK solver compatibility was fixed by switching from `daqp` to `quadprog`.
- Stable G1 visualization was exported in headless mode.
- The official annotated G1 locomanipulation dataset was downloaded.
- Mimic successfully generated G1 locomanipulation demonstrations.
- A generated demo can be replayed and exported as GIF/MP4 without robot flipping.
- A robomimic BC-RNN policy was trained with 1000 Mimic-generated demonstrations for 1800 epochs.
- The trained BC-RNN policy achieved 48/50 successful rollouts in a preliminary evaluation and a successful rollout GIF was exported.
- The same G1 Mimic-generated dataset was converted into a low-dimensional zarr dataset for Diffusion Policy training.
- A low-dimensional Diffusion Policy was trained and connected back to IsaacLab for closed-loop G1 rollout.
- A successful Diffusion Policy rollout GIF was exported.
- A critical deployment bug was fixed: the first Diffusion Policy rollout script incorrectly used `horizon=16` as the observation history length, while the policy was trained with `n_obs_steps=2`.

This repository currently focuses on **IsaacLab G1 locomanipulation reproduction, BC-RNN baseline training, and a low-dimensional Diffusion Policy extension**.

---

### Demo

#### robomimic BC-RNN policy rollout

The top GIF shows a successful rollout from the robomimic BC-RNN policy.

BC-RNN result:

- Dataset: 1000 Mimic-generated demonstrations.
- Training: 1800 epochs.
- Evaluation: 48/50 successful rollouts in a preliminary test.
- Task: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`.

#### Low-dimensional Diffusion Policy rollout

<p align="center">
  <img src="GIFs/diffusion_100.gif" width="720"/>
</p>

The GIF above shows a successful rollout from a low-dimensional Diffusion Policy trained on the same G1 Mimic-generated demonstrations.

Important implementation detail:

- The Diffusion Policy was trained with `n_obs_steps=2`, `n_action_steps=8`, and `horizon=16`.
- The first IsaacLab rollout script mistakenly used `horizon=16` as the observation history length.
- The corrected rollout uses `n_obs_steps=2` as the observation history length and executes the predicted action chunk with `exec_horizon`.
- This fixed the slow and unstable behavior observed in the initial rollout.

Generated demonstration replay was also verified. The correct replay path is to use raw 32-D `actions`, restore `initial_state`, and avoid feeding `processed_actions` into `env.step()`.

---

### Key Debugging Notes

#### 1. PickPlace task registration

The G1 pick-place task exists in the Isaac Lab source tree, but the corresponding pick-place modules may not be automatically imported by a normal task listing.

The scripts explicitly import:

```python
import isaaclab_tasks.manager_based.manipulation.pick_place
import isaaclab_tasks.manager_based.locomanipulation.pick_place
```

---

#### 2. Pinocchio / PinkIK initialization

This task depends on PinkIK and Pinocchio. In my environment, Pinocchio needs to be imported before launching Isaac Sim.

The custom scripts therefore support:

```bash
--enable_pinocchio
```

---

#### 3. PinkIK solver compatibility

The default PinkIK QP solver caused this error:

```text
solve() got an unexpected keyword argument 'primal_start'
```

The minimal fix was to switch the PinkIK solver from:

```python
solver="daqp"
```

to:

```python
solver="quadprog"
```

The patch note is stored in:

```text
patches/fix_pinkik_solver_quadprog.patch
```

---

#### 4. Generated demo replay

For generated G1 locomanipulation demos, replaying in the wrong environment or using `processed_actions` can cause the robot to flip.

The corrected replay path is:

- use the dataset's Mimic environment,
- restore `initial_state`,
- replay raw 32-D `actions`,
- avoid feeding `processed_actions` into `env.step()`.

---

#### 5. Diffusion Policy rollout alignment

For the low-dimensional Diffusion Policy rollout, the most important deployment issue was the difference between `horizon`, `n_obs_steps`, and `n_action_steps`.

The trained checkpoint used:

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8
```

The initial rollout script incorrectly used:

```text
obs_history_len = horizon = 16
```

The correct rollout setting is:

```text
obs_history_len = n_obs_steps = 2
```

Using the wrong observation history length caused the policy input distribution to differ from training, leading to slow and unstable motion. After this fix, the Diffusion Policy rollout became normal and could complete the task.

---

### Roadmap

- [x] Install Isaac Sim 5.1 and Isaac Lab on a headless server.
- [x] Register and launch the G1 fixed-base upper-body pick-place task.
- [x] Fix PinkIK solver compatibility.
- [x] Export stable G1 visualization video.
- [x] Download official annotated G1 locomanipulation dataset.
- [x] Generate G1 locomanipulation demonstrations with Mimic.
- [x] Replay generated demonstration without robot flipping.
- [x] Train a robomimic BC-RNN policy with 1000 Mimic-generated demonstrations.
- [x] Export successful trained BC-RNN policy rollout GIF.
- [x] Convert G1 Mimic demonstrations into a low-dimensional zarr dataset for Diffusion Policy.
- [x] Train and deploy a low-dimensional Diffusion Policy for IsaacLab G1 closed-loop rollout.
- [x] Export successful Diffusion Policy rollout GIF.

---

## 中文

### 项目简介

本仓库记录我对 **NVIDIA Isaac Lab 中 G1 pick-place / locomanipulation 相关任务** 的复现与调试过程。

目前包含三部分：

1. 复现和调试 fixed-base upper-body G1 pick-place 环境：

```text
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
```

2. 复现 Isaac Lab 官方 Mimic + robomimic 模仿学习流程中的 G1 locomanipulation 任务：

```text
Isaac-Locomanipulation-G1-Abs-Mimic-v0
Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

3. 将同一批 G1 Mimic 生成 demonstration 扩展到 low-dimensional Diffusion Policy 流程：

```text
G1 Mimic demonstrations
→ low-dimensional zarr dataset
→ Diffusion Policy training
→ IsaacLab closed-loop rollout
```

本项目基于 NVIDIA 官方 Isaac Lab：

```text
NVIDIA Isaac Lab
https://github.com/isaac-sim/IsaacLab
```

---

### 当前进度

目前已经验证：

- Isaac Sim 5.1 可以在无桌面的 RTX 4090 服务器上运行。
- Isaac Lab 源码环境可以正常启动。
- `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` 可以成功注册、reset 和 step。
- fixed-base upper-body 动作维度确认为 28。
- 已通过将 PinkIK solver 从 `daqp` 改为 `quadprog` 修复求解器兼容问题。
- 已在 headless 模式下稳定导出 G1 可视化结果。
- 已下载官方 G1 locomanipulation annotated dataset。
- 已使用 Mimic 成功生成 G1 locomanipulation demonstration。
- 已成功回放 generated demo，并导出 GIF/MP4，机器人不再乱翻滚。
- 已使用 1000 条 Mimic 生成 demonstration 训练 robomimic BC-RNN 策略 1800 epochs。
- BC-RNN 策略在 50 次初步 rollout 评估中成功 48 次，并已导出成功 rollout GIF。
- 已将同一批 G1 Mimic 生成 demonstration 转换为 low-dimensional zarr 数据集，用于 Diffusion Policy 训练。
- 已训练 low-dimensional Diffusion Policy，并接回 IsaacLab 完成 G1 闭环 rollout。
- 已导出 Diffusion Policy 成功 rollout GIF。
- 定位并修复 Diffusion Policy 部署中的关键问题：初版 rollout 误将 `horizon=16` 当作观测历史长度，而模型训练时实际使用的是 `n_obs_steps=2`。

当前仓库主要包括 **IsaacLab G1 locomanipulation 复现、BC-RNN baseline 训练，以及 low-dimensional Diffusion Policy 扩展实验**。

---

### 演示

#### robomimic BC-RNN 策略 rollout

顶部 GIF 展示的是训练后 robomimic BC-RNN 策略的一次成功 rollout。

BC-RNN 当前结果：

- 数据集：1000 条 Mimic 生成 demonstration。
- 训练轮数：1800 epochs。
- 初步评估：50 次 rollout 中成功 48 次。
- 任务：`Isaac-PickPlace-Locomanipulation-G1-Abs-v0`。

#### Low-dimensional Diffusion Policy rollout

<p align="center">
  <img src="GIFs/diffusion_100.gif" width="720"/>
</p>

上方 GIF 展示的是基于同一批 G1 Mimic demonstration 训练得到的 low-dimensional Diffusion Policy 的一次成功 rollout。

关键实现细节：

- Diffusion Policy 训练时使用 `n_obs_steps=2`、`n_action_steps=8`、`horizon=16`。
- 初版 IsaacLab rollout 脚本误将 `horizon=16` 当作观测历史长度。
- 修正后使用 `n_obs_steps=2` 作为观测历史长度，并通过 `exec_horizon` 执行预测出的 action chunk。
- 该修复解决了初版 rollout 中动作迟缓和抖动的问题。

此前 generated demonstration replay 也已验证。回放调试中确认，正确路径是使用 32 维原始 `actions`，恢复 `initial_state`，并避免将 `processed_actions` 直接传入 `env.step()`。

---

### 关键调试记录

#### 1. PickPlace 任务注册问题

G1 pick-place 任务本身存在于 Isaac Lab 源码中，但相关 pick-place 模块不一定会被普通任务列表自动导入。

因此脚本中需要显式导入：

```python
import isaaclab_tasks.manager_based.manipulation.pick_place
import isaaclab_tasks.manager_based.locomanipulation.pick_place
```

---

#### 2. Pinocchio / PinkIK 初始化问题

该任务依赖 PinkIK 和 Pinocchio。在当前环境中，需要在 Isaac Sim 启动前先导入 Pinocchio。

因此自定义脚本支持：

```bash
--enable_pinocchio
```

---

#### 3. PinkIK 求解器兼容问题

PinkIK 默认 QP solver 在当前环境中会报错：

```text
solve() got an unexpected keyword argument 'primal_start'
```

最小修复方式是把 PinkIK solver 从：

```python
solver="daqp"
```

改为：

```python
solver="quadprog"
```

补丁说明保存在：

```text
patches/fix_pinkik_solver_quadprog.patch
```

---

#### 4. Generated demo 回放问题

对于 generated G1 locomanipulation demo，如果使用错误环境回放，或者把 `processed_actions` 直接传给 `env.step()`，机器人会乱翻滚。

修正后的回放路径是：

- 使用数据集对应的 Mimic 环境；
- 恢复 `initial_state`；
- 回放 32 维原始 `actions`；
- 不把 `processed_actions` 传给 `env.step()`。

---

#### 5. Diffusion Policy rollout 观测历史长度错配

在 low-dimensional Diffusion Policy 接入 IsaacLab rollout 时，最关键的问题是区分 `horizon`、`n_obs_steps` 和 `n_action_steps`。

当前 checkpoint 使用：

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8
```

初版 rollout 脚本错误使用：

```text
obs_history_len = horizon = 16
```

正确做法应为：

```text
obs_history_len = n_obs_steps = 2
```

该错误会导致在线 rollout 时输入给 policy 的观测历史与训练时不一致，表现为动作迟缓、发钝和抖动。修正为 `n_obs_steps=2` 后，Diffusion Policy 能够正常完成任务。

---

### 后续计划

- [x] 在无桌面服务器上安装 Isaac Sim 5.1 和 Isaac Lab。
- [x] 注册并启动 G1 fixed-base upper-body pick-place 任务。
- [x] 修复 PinkIK solver 兼容问题。
- [x] 导出稳定的 G1 可视化视频。
- [x] 下载官方 G1 locomanipulation annotated dataset。
- [x] 使用 Mimic 生成 G1 locomanipulation demonstration。
- [x] 成功回放 generated demonstration，机器人不再乱翻滚。
- [x] 使用 1000 条 Mimic 生成 demonstration 训练 robomimic BC-RNN 策略。
- [x] 导出训练后 BC-RNN 策略成功 rollout GIF。
- [x] 将 G1 Mimic demonstration 转换为 low-dimensional zarr 数据集，用于 Diffusion Policy 训练。
- [x] 训练并部署 low-dimensional Diffusion Policy，完成 IsaacLab G1 闭环 rollout。
- [x] 导出 Diffusion Policy 成功 rollout GIF。