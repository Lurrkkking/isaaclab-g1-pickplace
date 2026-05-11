# IsaacLab G1 PickPlace

<p align="center">
  <a href="#english"><b>English</b></a> |
  <a href="#中文"><b>中文</b></a>
</p>

---

## English

### Overview

This repository records my reproduction and debugging process for the **G1 fixed-base upper-body pick-place task** in **NVIDIA Isaac Lab**.

The reproduced task is:

```text
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
```

This project is based on the official NVIDIA Isaac Lab project:

```text
NVIDIA Isaac Lab
https://github.com/isaac-sim/IsaacLab
```

This repository is **not a fork or mirror of Isaac Lab**. It only contains reproduction scripts, debugging notes, and minimal patches used to make the task run reliably on a headless RTX 4090 server.

---

### Current Status

Verified:

- Isaac Sim 5.1 runs on a headless RTX 4090 server.
- Isaac Lab is installed from source.
- The task `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` can be registered.
- The environment can be created, reset, and stepped.
- The action dimension is verified as 28.
- G1 robot visual is confirmed to exist in the USD stage.
- Viewport-based video export works.
- A hold-current-pose action is used for stable visualization.

This repository currently focuses on **environment reproduction and debugging**. It does not yet contain a trained pick-place policy.

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

The root cause was an incompatibility between `qpsolvers`, its DAQP adapter, and the installed `daqp` Python binding.

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

#### 4. Video export issue

Using Gymnasium `RecordVideo` with `render_mode="rgb_array"` did not reliably capture the full G1 robot visual.

The workaround is to use Isaac Sim viewport capture instead of the Gymnasium video wrapper.

---

#### 5. Zero action is not a no-op

This task uses an **absolute IK action space**. A zero action does not mean “hold current pose”.

For visualization, the script constructs a hold-current-pose action after reset:

- current left wrist world position and quaternion,
- current right wrist world position and quaternion,
- current hand joint positions.

This gives a valid 28-dimensional idle action and keeps the G1 robot visible during video export.

---

### Smoke Test

Run from the Isaac Lab root directory:

```bash
cd /root/autodl-tmp/IsaacLab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/isaacsim-5.1.0

unset LD_LIBRARY_PATH
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p -u scripts/g1_pickplace_smoke.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --num_envs 1 \
  --headless \
  --enable_pinocchio
```

Expected output:

```text
[INFO] reset ok
[INFO] action_dim: 28
[INFO] step 0 ok
[INFO] step 5 ok
[INFO] step 10 ok
[INFO] step 15 ok
```

---

### Video Export

```bash
cd /root/autodl-tmp/IsaacLab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/isaacsim-5.1.0

unset LD_LIBRARY_PATH
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p -u scripts/g1_pickplace_video.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --enable_pinocchio \
  --rendering_mode performance \
  --num_steps 10 \
  --video_length 10
```

---

### Repository Structure

```text
isaaclab-g1-pickplace/
├── README.md
├── scripts/
│   ├── g1_pickplace_smoke.py
│   └── g1_pickplace_video.py
├── patches/
│   └── fix_pinkik_solver_quadprog.patch
└── docs/
```

---

### Not Included

This repository does not include:

- Isaac Lab source code,
- Isaac Sim assets,
- USD assets,
- HDF5 demonstration datasets,
- checkpoints,
- generated videos,
- trained policies.

Large assets and generated outputs should be kept outside this repository.

---

### Roadmap

- [x] Install Isaac Sim 5.1 and Isaac Lab on a headless server.
- [x] Register and launch the G1 fixed-base upper-body pick-place task.
- [x] Fix PinkIK solver compatibility.
- [x] Export stable G1 visualization video.
- [ ] Analyze the 28-dimensional absolute IK action space.
- [ ] Reproduce the official demonstration / replay pipeline.
- [ ] Generate or replay HDF5 demonstration data.
- [ ] Train a robomimic behavior cloning policy.
- [ ] Export policy rollout videos.

---

## 中文

### 项目简介

本仓库记录我对 **NVIDIA Isaac Lab 官方 G1 fixed-base upper-body pick-place 任务** 的复现与调试过程。

复现目标任务为：

```text
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0
```

本项目基于 NVIDIA 官方 Isaac Lab：

```text
NVIDIA Isaac Lab
https://github.com/isaac-sim/IsaacLab
```

本仓库**不是 Isaac Lab 的 fork 或完整镜像**，只保存复现过程中使用的脚本、调试记录和少量补丁。

---

### 当前进度

目前已经验证：

- Isaac Sim 5.1 可以在无桌面的 RTX 4090 服务器上运行。
- Isaac Lab 源码环境可以正常启动。
- `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` 任务可以成功注册。
- 环境可以正常创建、reset 和 step。
- 动作维度确认为 28。
- G1 robot visual 已确认存在于 USD stage 中。
- 已实现稳定的 viewport 视频导出脚本。
- 已使用 hold-current-pose action 导出稳定的 G1 可视化视频。

当前仓库主要处于**环境复现与调试阶段**，还没有包含训练好的 pick-place 策略。

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

根因是 `qpsolvers` 的 DAQP adapter 和当前安装的 `daqp` Python binding 接口不兼容。

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

#### 4. 视频导出问题

使用 Gymnasium 的 `RecordVideo` 和 `render_mode="rgb_array"` 时，导出视频里不能稳定显示完整 G1 机器人。

解决方法是绕过 `RecordVideo`，改用 Isaac Sim 的 viewport capture 方式导出图像和视频。

---

#### 5. zero action 不是静止动作

该任务使用的是**绝对 IK 动作空间**，因此全 0 action 不是“不动”，而是会被解释为错误的绝对手腕目标和手指目标。

为了稳定可视化，视频脚本在 reset 后从当前机器人状态构造 hold-current-pose action：

- 左手腕当前世界坐标位置和四元数；
- 右手腕当前世界坐标位置和四元数；
- 当前手部关节位置。

这样可以得到一个合理的 28 维 idle action，使 G1 在视频中保持完整可见。

---

### Smoke Test

在 Isaac Lab 根目录运行：

```bash
cd /root/autodl-tmp/IsaacLab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/isaacsim-5.1.0

unset LD_LIBRARY_PATH
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p -u scripts/g1_pickplace_smoke.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --num_envs 1 \
  --headless \
  --enable_pinocchio
```

期望输出：

```text
[INFO] reset ok
[INFO] action_dim: 28
[INFO] step 0 ok
[INFO] step 5 ok
[INFO] step 10 ok
[INFO] step 15 ok
```

---

### 视频导出

```bash
cd /root/autodl-tmp/IsaacLab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/isaacsim-5.1.0

unset LD_LIBRARY_PATH
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p -u scripts/g1_pickplace_video.py \
  --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  --enable_pinocchio \
  --rendering_mode performance \
  --num_steps 10 \
  --video_length 10
```

---

### 仓库结构

```text
isaaclab-g1-pickplace/
├── README.md
├── scripts/
│   ├── g1_pickplace_smoke.py
│   └── g1_pickplace_video.py
├── patches/
│   └── fix_pinkik_solver_quadprog.patch
└── docs/
```

---

### 不包含的内容

本仓库不包含：

- Isaac Lab 源码；
- Isaac Sim 资产；
- USD 资产；
- HDF5 demonstration 数据；
- checkpoint；
- 生成的视频；
- 已训练策略。

大文件、资产和实验输出应放在仓库外部。

---

### 后续计划

- [x] 在无桌面服务器上安装 Isaac Sim 5.1 和 Isaac Lab。
- [x] 注册并启动 G1 fixed-base upper-body pick-place 任务。
- [x] 修复 PinkIK solver 兼容问题。
- [x] 导出稳定的 G1 可视化视频。
- [ ] 分析 28 维绝对 IK 动作空间。
- [ ] 复现官方 demonstration / replay 流程。
- [ ] 生成或回放 HDF5 demonstration 数据。
- [ ] 训练 robomimic 行为克隆策略。
- [ ] 导出策略 rollout 视频。