#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import pathlib
import random
import sys
import traceback
import types
from typing import Any, Dict, Optional

import tomllib

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch


if "toml" not in sys.modules:
    toml_module = types.ModuleType("toml")

    def _toml_load(obj, *args, **kwargs):
        if hasattr(obj, "read"):
            data = obj.read()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return tomllib.loads(data)
        with open(obj, "rb") as f:
            return tomllib.load(f)

    def _toml_loads(s, *args, **kwargs):
        return tomllib.loads(s)

    toml_module.load = _toml_load
    toml_module.loads = _toml_loads
    sys.modules["toml"] = toml_module


OBS_KEYS = [
    "left_eef_pos",
    "left_eef_quat",
    "right_eef_pos",
    "right_eef_quat",
    "hand_joint_state",
    "object",
]

PERTURB_CONFIGS = {
    "none": {"xy_max": 0.0, "yaw_deg_max": 0.0},
    "mild": {"xy_max": 0.02, "yaw_deg_max": 5.0},
    "medium": {"xy_max": 0.05, "yaw_deg_max": 15.0},
    "hard": {"xy_max": 0.08, "yaw_deg_max": 30.0},
}


parser = argparse.ArgumentParser(description="Minimal IsaacLab rollout for G1 low-dimensional ACT policy.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/root/autodl-tmp/act_g1/outputs/g1_act_smoke/checkpoints/latest.pt",
)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--horizon", type=int, default=50)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--record_video", action="store_true", default=False)
parser.add_argument("--video_dir", type=str, default="videos/act_rollout")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--verbose_steps", action="store_true", default=False)
parser.add_argument("--debug_env_only", action="store_true", default=False)
parser.add_argument("--debug_action_trace", action="store_true", default=False)
parser.add_argument("--perturb_level", type=str, choices=["none", "mild", "medium", "hard"], default="none")
parser.add_argument("--obs_noise_std", type=float, default=0.0)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--action_noise_mode", type=str, choices=["none", "iid", "smooth"], default="none")
parser.add_argument("--action_noise_beta", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_failure_log", action="store_true", default=False)
parser.add_argument("--failure_log_path", type=str, default="logs/act_rollout_failure_log.json")

args_pre, _ = parser.parse_known_args()
if args_pre.enable_pinocchio:
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] App launched", flush=True)

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401,E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
print("[INFO] IsaacLab task package imported", flush=True)


def ensure_dir(path_str: str) -> None:
    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)


def nested_keys(obj) -> list[str]:
    if hasattr(obj, "keys"):
        return sorted(str(key) for key in obj.keys())
    return [str(type(obj))]


def tensor_flag_to_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def unwrap_policy_obs(obs):
    if not hasattr(obs, "keys"):
        raise TypeError(f"Observation container has no keys: {type(obs)}")
    if "policy" in obs:
        return obs["policy"]
    return obs


def flatten_policy_obs(policy_obs) -> torch.Tensor:
    if not hasattr(policy_obs, "keys"):
        raise TypeError(f"Policy observation container has no keys: {type(policy_obs)}")

    flat_parts = []
    for key in OBS_KEYS:
        if key not in policy_obs:
            raise KeyError(f"Missing observation key '{key}'. Available keys: {nested_keys(policy_obs)}")
        value = policy_obs[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        elif value.ndim != 2:
            raise ValueError(f"Unexpected shape for obs key '{key}': {tuple(value.shape)}")
        flat_parts.append(value)

    flat_obs = torch.cat(flat_parts, dim=-1)
    if flat_obs.shape[-1] != 41:
        raise ValueError(f"Flat obs last dim should be 41, got {tuple(flat_obs.shape)}")
    return flat_obs


def set_camera(env) -> None:
    env.unwrapped.sim.set_camera_view(
        eye=[3.0, 4.0, 2.2],
        target=[0.0, 0.2, 0.9],
    )


def tensor_stats_str(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float().cpu()
    return (
        f"shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f}"
    )


def tensor_to_list(value) -> Optional[list[float]]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().float().cpu()
    if value.ndim > 1:
        value = value[0]
    return [float(x) for x in value.tolist()]


def extract_policy_state(policy_obs) -> dict[str, Optional[list[float]]]:
    return {
        "object_pos": tensor_to_list(policy_obs.get("object_pos")),
        "left_eef_pos": tensor_to_list(policy_obs.get("left_eef_pos")),
        "right_eef_pos": tensor_to_list(policy_obs.get("right_eef_pos")),
    }


def perturb_spec(level: str) -> dict[str, float]:
    if level not in PERTURB_CONFIGS:
        raise ValueError(f"Unsupported perturb_level: {level}")
    return dict(PERTURB_CONFIGS[level])


def sample_rollout_perturbation(level: str, rng: np.random.Generator) -> dict[str, Any]:
    spec = perturb_spec(level)
    xy_max = float(spec["xy_max"])
    yaw_deg_max = float(spec["yaw_deg_max"])
    xy_offset = np.zeros(2, dtype=np.float32)
    yaw_deg = 0.0
    if xy_max > 0.0:
        xy_offset = rng.uniform(low=-xy_max, high=xy_max, size=2).astype(np.float32)
    if yaw_deg_max > 0.0:
        yaw_deg = float(rng.uniform(low=-yaw_deg_max, high=yaw_deg_max))
    return {
        "level": level,
        "xy_offset": xy_offset,
        "yaw_deg": yaw_deg,
        "yaw_rad": math.radians(yaw_deg),
    }


def apply_object_pose_perturbation(env, perturbation: dict[str, Any]) -> tuple[bool, Optional[str]]:
    if perturbation["level"] == "none":
        return True, None

    xy_offset = perturbation["xy_offset"]
    yaw_rad = float(perturbation["yaw_rad"])
    if float(np.linalg.norm(xy_offset)) == 0.0 and yaw_rad == 0.0:
        return True, None

    try:
        scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
        try:
            rigid_object = scene["object"]
        except Exception as exc:
            return False, f"failed to access scene['object']: {exc}"
        root_state_w = rigid_object.data.root_state_w.clone()
        env_ids = torch.tensor([0], dtype=torch.int64, device=root_state_w.device)
        updated_root_state_w = root_state_w[env_ids].clone()
        updated_root_state_w[:, 0] += float(xy_offset[0])
        updated_root_state_w[:, 1] += float(xy_offset[1])

        yaw_tensor = torch.full((len(env_ids),), yaw_rad, dtype=updated_root_state_w.dtype, device=root_state_w.device)
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_tensor),
            torch.zeros_like(yaw_tensor),
            yaw_tensor,
        )
        updated_root_state_w[:, 3:7] = math_utils.quat_mul(updated_root_state_w[:, 3:7], yaw_quat)
        updated_root_state_w[:, 7:] = 0.0

        rigid_object.write_root_pose_to_sim(updated_root_state_w[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(updated_root_state_w[:, 7:], env_ids=env_ids)
        scene.write_data_to_sim()
        env.sim.forward()
        scene.update(0.0)
        return True, None
    except Exception as exc:
        return False, str(exc)


def refresh_obs_after_perturbation(env):
    env.obs_buf = env.observation_manager.compute(update_history=False)
    return env.obs_buf, env.extras


def maybe_add_obs_noise(flat_obs: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0.0:
        return flat_obs
    return flat_obs + torch.randn_like(flat_obs) * noise_std


class ActionNoiseState:
    def __init__(self, mode: str, noise_std: float, beta: float):
        self.mode = mode
        self.noise_std = noise_std
        self.beta = beta
        self.prev_noise: Optional[torch.Tensor] = None

    def apply(self, action: torch.Tensor) -> torch.Tensor:
        if self.mode == "none" or self.noise_std <= 0.0:
            return action
        if self.mode == "iid":
            return action + torch.randn_like(action) * self.noise_std
        if self.mode == "smooth":
            eps = torch.randn_like(action) * self.noise_std
            if self.prev_noise is None:
                noise = (1.0 - self.beta) * eps
            else:
                noise = self.beta * self.prev_noise + (1.0 - self.beta) * eps
            self.prev_noise = noise
            return action + noise
        raise ValueError(f"Unsupported action_noise_mode: {self.mode}")


def infer_failure_reason(success: bool, terminated: bool, truncated: bool, end_step: Optional[int], horizon: int) -> str:
    if success:
        return "success"
    if terminated:
        return "terminated_failed"
    if truncated:
        return "truncated_failed"
    if end_step is not None and end_step >= horizon - 1:
        return "timeout"
    return "timeout"


def save_failure_log(records: list[dict[str, Any]], log_path: str) -> None:
    path = pathlib.Path(log_path)
    ensure_dir(str(path.parent))

    if path.suffix.lower() == ".csv":
        fieldnames = [
            "rollout_index",
            "success",
            "success_step",
            "end_step",
            "perturb_level",
            "object_xy_range",
            "object_yaw_range",
            "object_xy_offset",
            "object_yaw_offset",
            "obs_noise_std",
            "action_noise_std",
            "action_noise_mode",
            "final_object_pos",
            "final_left_eef_pos",
            "final_right_eef_pos",
            "failure_reason",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "rollout_index": record["rollout_index"],
                        "success": record["success"],
                        "success_step": record["success_step"],
                        "end_step": record["end_step"],
                        "perturb_level": record["perturb_level"],
                        "object_xy_range": record["object_xy_range"],
                        "object_yaw_range": record["object_yaw_range"],
                        "object_xy_offset": json.dumps(record["object_xy_offset"]),
                        "object_yaw_offset": record["object_yaw_offset"],
                        "obs_noise_std": record["obs_noise_std"],
                        "action_noise_std": record["action_noise_std"],
                        "action_noise_mode": record["action_noise_mode"],
                        "final_object_pos": json.dumps(record["final_object_pos"]),
                        "final_left_eef_pos": json.dumps(record["final_left_eef_pos"]),
                        "final_right_eef_pos": json.dumps(record["final_right_eef_pos"]),
                        "failure_reason": record["failure_reason"],
                    }
                )
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_torch_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


class ACTNormalizer:
    def __init__(self, stats: Dict[str, np.ndarray], device: torch.device):
        self.obs_mean = torch.as_tensor(stats["obs_mean"], dtype=torch.float32, device=device)
        self.obs_std = torch.as_tensor(stats["obs_std"], dtype=torch.float32, device=device)
        self.action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=device)
        self.action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=device)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean


def import_act_model_class():
    act_root = "/root/autodl-tmp/act_g1"
    if act_root not in sys.path:
        sys.path.insert(0, act_root)
    module = importlib.import_module("act_g1.model")
    return module.MinimalACTModel


def load_policy(checkpoint_path: str, device: torch.device):
    ckpt_path = pathlib.Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = load_torch_checkpoint(str(ckpt_path), map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    dataset_config = dict(checkpoint.get("dataset_config", {}))
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        raise KeyError("Checkpoint is missing normalization_stats")

    model_cls = import_act_model_class()
    model = model_cls(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer = ACTNormalizer(stats=stats, device=device)
    return model, normalizer, {
        "model_config": model_config,
        "dataset_config": dataset_config,
        "epoch": checkpoint.get("epoch"),
    }


class RolloutVideoRecorder:
    def __init__(self, env, video_dir: str, rollout_idx: int, fps: int, enabled: bool):
        self.env = env
        self.enabled = enabled
        self.fps = fps
        self.video_dir = pathlib.Path(video_dir)
        self.rollout_idx = rollout_idx
        self.video_path = self.video_dir / f"rollout_{rollout_idx}.mp4"
        self.mode = None
        self.writer = None

    def start(self):
        if not self.enabled:
            return
        ensure_dir(str(self.video_dir))

        frame = self._safe_render()
        if frame is None:
            raise RuntimeError("Video recording requires env.render() to return rgb_array frames.")
        self.mode = "rgb_array"
        self.writer = imageio.get_writer(str(self.video_path), fps=self.fps)
        self.writer.append_data(frame)
        print(f"[INFO] video recorder mode=rgb_array path={self.video_path}", flush=True)

    def capture_step(self):
        if not self.enabled:
            return
        if self.mode != "rgb_array":
            raise RuntimeError(f"Unsupported recorder mode: {self.mode}")
        frame = self._safe_render()
        if frame is None:
            raise RuntimeError("env.render() returned None while rgb_array recording was active.")
        self.writer.append_data(frame)

    def finish(self):
        if not self.enabled:
            return
        if self.mode == "rgb_array" and self.writer is not None:
            self.writer.close()
            return

    def _safe_render(self):
        try:
            return self.env.render()
        except NotImplementedError as exc:
            raise RuntimeError(
                "Video recording requires gym.make(..., render_mode='rgb_array'). "
                f"Current render path is unsupported: {exc}"
            ) from exc


class ActionTraceRecorder:
    def __init__(self, output_dir: str, rollout_idx: int, enabled: bool):
        self.enabled = enabled
        self.output_dir = pathlib.Path(output_dir)
        self.rollout_idx = rollout_idx
        self.path = self.output_dir / f"rollout_{rollout_idx}_action_trace.jsonl"
        self.records = 0

    def start(self):
        if not self.enabled:
            return
        ensure_dir(str(self.output_dir))
        if self.path.exists():
            self.path.unlink()

    def record(
        self,
        step_idx: int,
        env_action: torch.Tensor,
        prev_env_action: Optional[torch.Tensor],
        policy_obs,
    ) -> None:
        if not self.enabled:
            return
        action_cpu = env_action.detach().float().cpu()
        if prev_env_action is None:
            action_delta_mean_abs = 0.0
        else:
            action_delta_mean_abs = (
                action_cpu - prev_env_action.detach().float().cpu()
            ).abs().mean().item()

        def _get_vec(key: str):
            value = policy_obs.get(key)
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value[0].detach().float().cpu().tolist()
            return value

        record = {
            "step_idx": int(step_idx),
            "action_min": float(action_cpu.min().item()),
            "action_max": float(action_cpu.max().item()),
            "action_mean": float(action_cpu.mean().item()),
            "action_delta_mean_abs": float(action_delta_mean_abs),
            "left_eef_pos": _get_vec("left_eef_pos"),
            "right_eef_pos": _get_vec("right_eef_pos"),
            "object_pos": _get_vec("object_pos"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records += 1

    def finish(self):
        if self.enabled:
            print(f"[INFO] debug_action_trace path={self.path}", flush=True)


def maybe_render_mode() -> Optional[str]:
    if args_cli.record_video:
        return "rgb_array"
    return None


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    if args_cli.exec_horizon < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {args_cli.exec_horizon}")
    if args_cli.obs_noise_std < 0.0:
        raise ValueError(f"obs_noise_std must be >= 0, got {args_cli.obs_noise_std}")
    if args_cli.action_noise_std < 0.0:
        raise ValueError(f"action_noise_std must be >= 0, got {args_cli.action_noise_std}")
    if not (0.0 <= args_cli.action_noise_beta < 1.0):
        raise ValueError(f"action_noise_beta must be in [0, 1), got {args_cli.action_noise_beta}")

    device = torch.device(args_cli.device)
    perturbation_rng = np.random.default_rng(args_cli.seed)
    perturbation_cfg = perturb_spec(args_cli.perturb_level)
    print("[INFO] Before parse_env_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    print("[INFO] After parse_env_cfg", flush=True)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.viewer.eye = (3.0, 4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.9)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None

    print("[INFO] Before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=maybe_render_mode()).unwrapped
    print("[INFO] Env created", flush=True)
    env.seed(args_cli.seed)

    print(
        "[INFO] Evaluation perturbation settings: "
        f"level={args_cli.perturb_level} "
        f"object_xy_range={perturbation_cfg['xy_max']} "
        f"object_yaw_range_deg={perturbation_cfg['yaw_deg_max']} "
        f"obs_noise_std={args_cli.obs_noise_std} "
        f"action_noise_std={args_cli.action_noise_std} "
        f"action_noise_mode={args_cli.action_noise_mode} "
        f"action_noise_beta={args_cli.action_noise_beta} "
        f"seed={args_cli.seed}",
        flush=True,
    )

    print("[INFO] Before loading ACT", flush=True)
    model, normalizer, ckpt_meta = load_policy(args_cli.checkpoint, device=device)
    print("[INFO] ACT policy loaded", flush=True)
    model_config = ckpt_meta["model_config"]
    n_obs_steps = int(model_config["n_obs_steps"])
    chunk_size = int(model_config["chunk_size"])
    action_dim = int(model_config["action_dim"])
    obs_dim = int(model_config["obs_dim"])
    if n_obs_steps != 2:
        raise ValueError(f"ACT rollout expects n_obs_steps=2, got checkpoint value {n_obs_steps}")
    if chunk_size != 8:
        raise ValueError(f"ACT rollout expects chunk_size=8, got checkpoint value {chunk_size}")
    if obs_dim != 41:
        raise ValueError(f"ACT rollout expects obs_dim=41, got checkpoint value {obs_dim}")
    if action_dim != 32:
        raise ValueError(f"ACT rollout expects action_dim=32, got checkpoint value {action_dim}")

    print(f"checkpoint path: {args_cli.checkpoint}", flush=True)
    print(f"env action_space: {env.action_space}", flush=True)
    print(
        "[INFO] ACT checkpoint params: "
        f"epoch={ckpt_meta['epoch']} "
        f"n_obs_steps={n_obs_steps} "
        f"chunk_size={chunk_size} "
        f"obs_dim={obs_dim} "
        f"action_dim={action_dim}",
        flush=True,
    )

    trial_results = []
    failure_records = []
    pose_perturb_warned = False
    try:
        for rollout_idx in range(args_cli.num_rollouts):
            print("[INFO] Before env.reset", flush=True)
            obs, info = env.reset()
            print("[INFO] reset ok", flush=True)
            set_camera(env)

            rollout_perturbation = sample_rollout_perturbation(args_cli.perturb_level, perturbation_rng)
            pose_perturb_applied = True
            pose_perturb_warning = None
            if args_cli.perturb_level != "none":
                pose_perturb_applied, pose_perturb_warning = apply_object_pose_perturbation(env, rollout_perturbation)
                if pose_perturb_applied:
                    obs, info = refresh_obs_after_perturbation(env)
                else:
                    if not pose_perturb_warned:
                        print(
                            "[WARNING] Object pose perturbation is disabled for this run because applying it failed: "
                            f"{pose_perturb_warning}",
                            flush=True,
                        )
                        pose_perturb_warned = True
                    rollout_perturbation["xy_offset"] = np.zeros(2, dtype=np.float32)
                    rollout_perturbation["yaw_deg"] = 0.0
                    rollout_perturbation["yaw_rad"] = 0.0

            if args_cli.debug_env_only:
                debug_action = torch.zeros((1, env.action_space.shape[-1]), dtype=torch.float32, device=env.device)
                obs, reward, terminated, truncated, info = env.step(debug_action)
                print("[INFO] first env.step ok", flush=True)
                print(
                    f"rollout {rollout_idx} summary: success=False, success_step=None, "
                    f"end_step=0, terminated={tensor_flag_to_bool(terminated)}, "
                    f"truncated={tensor_flag_to_bool(truncated)}",
                    flush=True,
                )
                trial_results.append(False)
                continue

            video_recorder = RolloutVideoRecorder(
                env=env,
                video_dir=args_cli.video_dir,
                rollout_idx=rollout_idx,
                fps=args_cli.fps,
                enabled=args_cli.record_video,
            )
            action_trace_recorder = ActionTraceRecorder(
                output_dir=args_cli.video_dir,
                rollout_idx=rollout_idx,
                enabled=args_cli.debug_action_trace,
            )
            video_recorder.start()
            action_trace_recorder.start()

            policy_obs = unwrap_policy_obs(obs)
            flat_obs = maybe_add_obs_noise(flat_obs=flatten_policy_obs(policy_obs).to(device), noise_std=args_cli.obs_noise_std)
            obs_history = flat_obs.unsqueeze(1).repeat(1, n_obs_steps, 1)
            if rollout_idx == 0:
                print(f"[INFO] initial flat obs: {tensor_stats_str(flat_obs)}", flush=True)
                print(f"[INFO] initial obs_history shape: {tuple(obs_history.shape)}", flush=True)

            success_step = None
            end_step = None
            rollout_success = False
            rollout_terminated = False
            rollout_truncated = False
            env_step_idx = 0
            prev_env_action = None
            last_policy_obs = policy_obs
            last_step_idx = None
            action_noise_state = ActionNoiseState(
                mode=args_cli.action_noise_mode,
                noise_std=float(args_cli.action_noise_std),
                beta=float(args_cli.action_noise_beta),
            )

            while env_step_idx < args_cli.horizon:
                with torch.no_grad():
                    if rollout_idx == 0 and env_step_idx == 0:
                        print("[INFO] Before first model forward", flush=True)
                    normalized_obs_history = normalizer.normalize_obs(obs_history)
                    action_chunk_norm = model(normalized_obs_history)
                    action_chunk_raw = normalizer.denormalize_action(action_chunk_norm)

                if action_chunk_norm.shape != (1, chunk_size, action_dim):
                    raise ValueError(
                        f"Expected action_chunk shape (1, {chunk_size}, {action_dim}), "
                        f"got {tuple(action_chunk_norm.shape)}"
                    )
                exec_horizon = min(args_cli.exec_horizon, chunk_size, args_cli.horizon - env_step_idx)
                if rollout_idx == 0 and env_step_idx == 0:
                    print(f"[INFO] first action_chunk shape = {tuple(action_chunk_norm.shape)}", flush=True)
                    print(f"[INFO] ACT raw action_chunk stats: {tensor_stats_str(action_chunk_raw)}", flush=True)

                for exec_idx in range(exec_horizon):
                    env_action = action_chunk_raw[:, exec_idx, :].to(env.device)
                    env_action = action_noise_state.apply(env_action)
                    if env_action.ndim != 2 or env_action.shape[1] != 32:
                        raise ValueError(f"env.step action must be (num_envs, 32), got {tuple(env_action.shape)}")
                    if args_cli.verbose_steps:
                        print(
                            f"rollout {rollout_idx} env_step {env_step_idx} exec_idx {exec_idx} "
                            f"action_stats={tensor_stats_str(env_action)}"
                        )

                    obs, reward, terminated, truncated, info = env.step(env_action)
                    if rollout_idx == 0 and env_step_idx == 0 and exec_idx == 0:
                        print("[INFO] first env.step ok", flush=True)
                    video_recorder.capture_step()

                    policy_obs = unwrap_policy_obs(obs)
                    action_trace_recorder.record(
                        step_idx=env_step_idx,
                        env_action=env_action,
                        prev_env_action=prev_env_action,
                        policy_obs=policy_obs,
                    )
                    prev_env_action = env_action.clone()
                    flat_obs = maybe_add_obs_noise(
                        flat_obs=flatten_policy_obs(policy_obs).to(device),
                        noise_std=args_cli.obs_noise_std,
                    )
                    obs_history = torch.cat([obs_history[:, 1:, :], flat_obs.unsqueeze(1)], dim=1)
                    last_policy_obs = policy_obs
                    last_step_idx = env_step_idx

                    success = False
                    if success_term is not None:
                        success = tensor_flag_to_bool(success_term.func(env, **success_term.params))
                    terminated_flag = tensor_flag_to_bool(terminated)
                    truncated_flag = tensor_flag_to_bool(truncated)
                    done = terminated_flag or truncated_flag

                    if args_cli.verbose_steps:
                        reward_value = float(reward.reshape(-1)[0]) if isinstance(reward, torch.Tensor) else float(reward)
                        print(
                            f"rollout {rollout_idx} env_step {env_step_idx}: "
                            f"reward={reward_value:.6f} terminated={terminated_flag} "
                            f"truncated={truncated_flag} success={success}"
                        )

                    if success:
                        success_step = env_step_idx
                        end_step = env_step_idx
                        rollout_success = True
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        break

                    if done:
                        end_step = env_step_idx
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        break

                    env_step_idx += 1

                if rollout_success or rollout_terminated or rollout_truncated:
                    break

            if end_step is None:
                end_step = last_step_idx
            video_recorder.finish()
            action_trace_recorder.finish()
            trial_results.append(rollout_success)
            final_state = extract_policy_state(last_policy_obs)
            failure_reason = infer_failure_reason(
                success=rollout_success,
                terminated=rollout_terminated,
                truncated=rollout_truncated,
                end_step=end_step,
                horizon=args_cli.horizon,
            )
            if not pose_perturb_applied and pose_perturb_warning is not None:
                failure_reason = f"{failure_reason}; object_pose_perturbation_disabled"
            failure_record = {
                "rollout_index": rollout_idx,
                "success": rollout_success,
                "success_step": success_step,
                "end_step": end_step,
                "perturb_level": args_cli.perturb_level,
                "object_xy_range": float(perturbation_cfg["xy_max"]),
                "object_yaw_range": float(perturbation_cfg["yaw_deg_max"]),
                "object_xy_offset": [float(x) for x in rollout_perturbation["xy_offset"].tolist()],
                "object_yaw_offset": float(rollout_perturbation["yaw_deg"]),
                "obs_noise_std": float(args_cli.obs_noise_std),
                "action_noise_std": float(args_cli.action_noise_std),
                "action_noise_mode": args_cli.action_noise_mode,
                "final_object_pos": final_state["object_pos"],
                "final_left_eef_pos": final_state["left_eef_pos"],
                "final_right_eef_pos": final_state["right_eef_pos"],
                "failure_reason": failure_reason,
            }
            failure_records.append(failure_record)
            print(
                f"rollout {rollout_idx} summary: success={rollout_success}, "
                f"success_step={success_step}, end_step={end_step}, "
                f"terminated={rollout_terminated}, truncated={rollout_truncated}, "
                f"perturb_level={args_cli.perturb_level}, "
                f"object_xy_range={perturbation_cfg['xy_max']:.3f}, "
                f"object_yaw_range_deg={perturbation_cfg['yaw_deg_max']:.1f}, "
                f"object_xy_offset={[round(x, 4) for x in failure_record['object_xy_offset']]}, "
                f"object_yaw_offset_deg={failure_record['object_yaw_offset']:.3f}, "
                f"obs_noise_std={args_cli.obs_noise_std}, "
                f"action_noise_std={args_cli.action_noise_std}, "
                f"action_noise_mode={args_cli.action_noise_mode}, "
                f"failure_reason={failure_reason}",
                flush=True,
            )
        success_count = sum(trial_results)
        success_steps = [record["success_step"] for record in failure_records if record["success_step"] is not None]
        avg_success_step = float(np.mean(success_steps)) if success_steps else None
        if args_cli.save_failure_log:
            save_failure_log(failure_records, args_cli.failure_log_path)
        print(f"Successful trials: {success_count} / {len(trial_results)}", flush=True)
        print(f"Success rate: {success_count / len(trial_results):.4f}", flush=True)
        print(f"Avg success step if any: {avg_success_step}", flush=True)
        print(f"Perturb level: {args_cli.perturb_level}", flush=True)
        print(f"Object xy range: {perturbation_cfg['xy_max']}", flush=True)
        print(f"Object yaw range: {perturbation_cfg['yaw_deg_max']} deg", flush=True)
        print(
            f"Noise settings: obs_noise_std={args_cli.obs_noise_std}, "
            f"action_noise_std={args_cli.action_noise_std}, "
            f"action_noise_mode={args_cli.action_noise_mode}",
            flush=True,
        )
        print(
            f"Failure log path: {args_cli.failure_log_path if args_cli.save_failure_log else None}",
            flush=True,
        )
    finally:
        print("[INFO] Before env cleanup", flush=True)
        print("[INFO] Skipping env.close(); relying on simulation_app.close() for shutdown", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        print("[INFO] Before simulation_app.close()", flush=True)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        print("[INFO] simulation_app.close() returned", flush=True)
