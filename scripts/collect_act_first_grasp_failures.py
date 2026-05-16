#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import pickle
import random
import sys
import traceback
import types
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import tomllib


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


parser = argparse.ArgumentParser(description="Collect ACT first-grasp failures for G1 locomanipulation.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--horizon", type=int, default=400)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--perturb_level", type=str, choices=["none", "mild", "medium", "hard"], default="hard")
parser.add_argument("--obs_noise_std", type=float, default=0.0)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--window_before", type=int, default=12)
parser.add_argument("--window_after", type=int, default=36)

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

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def ensure_dir(path_str: str) -> None:
    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)


def tensor_flag_to_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def unwrap_policy_obs(obs):
    if "policy" in obs:
        return obs["policy"]
    return obs


def flatten_policy_obs(policy_obs) -> torch.Tensor:
    flat_parts = []
    for key in OBS_KEYS:
        value = policy_obs[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        flat_parts.append(value)
    flat_obs = torch.cat(flat_parts, dim=-1)
    if flat_obs.shape[-1] != 41:
        raise ValueError(f"Flat obs dim must be 41, got {tuple(flat_obs.shape)}")
    return flat_obs


def set_camera(env) -> None:
    env.unwrapped.sim.set_camera_view(eye=[3.0, 4.0, 2.2], target=[0.0, 0.2, 0.9])


def perturb_spec(level: str) -> dict[str, float]:
    if level not in PERTURB_CONFIGS:
        raise ValueError(f"Unsupported perturb level: {level}")
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
        rigid_object = scene["object"]
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


def load_torch_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(checkpoint_path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


class ACTNormalizer:
    def __init__(self, stats: dict[str, np.ndarray], device: torch.device):
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
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    model_cls = import_act_model_class()
    model = model_cls(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    normalizer = ACTNormalizer(stats=checkpoint["normalization_stats"], device=device)
    return model, normalizer, model_config, checkpoint.get("epoch")


def maybe_add_obs_noise(flat_obs: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0.0:
        return flat_obs
    return flat_obs + torch.randn_like(flat_obs) * noise_std


def to_numpy_1d(value, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=dtype)
    if value.ndim > 1:
        value = value[0]
    return value.astype(dtype, copy=False)


def get_object_pos(policy_obs) -> np.ndarray:
    if "object_pos" in policy_obs:
        return to_numpy_1d(policy_obs["object_pos"])
    return to_numpy_1d(policy_obs["object"])[:3]


def get_object_rot(env, policy_obs) -> np.ndarray:
    if "object_rot" in policy_obs:
        return to_numpy_1d(policy_obs["object_rot"])
    try:
        scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
        rigid_object = scene["object"]
        return to_numpy_1d(rigid_object.data.root_state_w[0, 3:7])
    except Exception:
        return np.zeros((4,), dtype=np.float32)


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


@dataclass
class RolloutMetrics:
    left_dist: np.ndarray
    right_dist: np.ndarray
    object_pos: np.ndarray
    object_rot: np.ndarray
    left_eef_pos: np.ndarray
    right_eef_pos: np.ndarray
    hand_joint_state: np.ndarray
    flat_obs: np.ndarray
    raw_action: np.ndarray
    step_indices: np.ndarray
    min_left_dist: float
    min_left_idx: int
    final_left_dist: float
    object_motion_range: float


def compute_object_motion_range(object_pos: np.ndarray) -> float:
    if len(object_pos) <= 1:
        return 0.0
    disp = np.linalg.norm(object_pos - object_pos[0:1], axis=1)
    return float(disp.max())


def compute_metrics(step_records: list[dict[str, Any]]) -> RolloutMetrics:
    object_pos = np.asarray([r["object_pos"] for r in step_records], dtype=np.float32)
    object_rot = np.asarray([r["object_rot"] for r in step_records], dtype=np.float32)
    left_eef_pos = np.asarray([r["left_eef_pos"] for r in step_records], dtype=np.float32)
    right_eef_pos = np.asarray([r["right_eef_pos"] for r in step_records], dtype=np.float32)
    hand_joint_state = np.asarray([r["hand_joint_state"] for r in step_records], dtype=np.float32)
    flat_obs = np.asarray([r["flat_obs"] for r in step_records], dtype=np.float32)
    raw_action = np.asarray([r["raw_action"] for r in step_records], dtype=np.float32)
    step_indices = np.asarray([r["step_index"] for r in step_records], dtype=np.int64)
    left_dist = np.linalg.norm(object_pos - left_eef_pos, axis=1)
    right_dist = np.linalg.norm(object_pos - right_eef_pos, axis=1)
    min_left_idx = int(np.argmin(left_dist))
    return RolloutMetrics(
        left_dist=left_dist.astype(np.float32, copy=False),
        right_dist=right_dist.astype(np.float32, copy=False),
        object_pos=object_pos,
        object_rot=object_rot,
        left_eef_pos=left_eef_pos,
        right_eef_pos=right_eef_pos,
        hand_joint_state=hand_joint_state,
        flat_obs=flat_obs,
        raw_action=raw_action,
        step_indices=step_indices,
        min_left_dist=float(left_dist[min_left_idx]),
        min_left_idx=min_left_idx,
        final_left_dist=float(left_dist[-1]),
        object_motion_range=compute_object_motion_range(object_pos),
    )


def classify_failure(metrics: RolloutMetrics, success: bool) -> tuple[str, str, bool]:
    if success:
        return "success", "success", False
    if metrics.min_left_dist > 0.15 and metrics.object_motion_range <= 0.03:
        return "pure_approach_failed", "pure_approach_failed", False
    if (
        metrics.min_left_dist <= 0.12
        and metrics.object_motion_range > 0.03
    ) or (
        metrics.min_left_dist <= 0.08
        and metrics.final_left_dist > max(0.14, metrics.min_left_dist + 0.04)
    ):
        return "first_grasp_failed", "first_grasp_failed", True
    if metrics.object_motion_range > 0.03:
        return "transport_failed", "transport_failed", False
    return "unknown_failed", "unknown_failed", False


def build_step_record(
    env,
    policy_obs,
    flat_obs: np.ndarray,
    raw_action: np.ndarray,
    step_index: int,
) -> dict[str, Any]:
    return {
        "step_index": int(step_index),
        "flat_obs": flat_obs.astype(np.float32, copy=True),
        "raw_action": raw_action.astype(np.float32, copy=True),
        "object_pos": get_object_pos(policy_obs).astype(np.float32, copy=True),
        "object_rot": get_object_rot(env, policy_obs).astype(np.float32, copy=True),
        "left_eef_pos": to_numpy_1d(policy_obs["left_eef_pos"]).astype(np.float32, copy=True),
        "right_eef_pos": to_numpy_1d(policy_obs["right_eef_pos"]).astype(np.float32, copy=True),
        "hand_joint_state": to_numpy_1d(policy_obs["hand_joint_state"]).astype(np.float32, copy=True),
    }


def save_pickle(path: pathlib.Path, payload: Any) -> None:
    ensure_dir(str(path.parent))
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    if args_cli.exec_horizon < 1:
        raise ValueError("exec_horizon must be >= 1")
    if args_cli.obs_noise_std < 0.0 or args_cli.action_noise_std < 0.0:
        raise ValueError("noise std must be >= 0")

    save_dir = pathlib.Path(args_cli.save_dir)
    ensure_dir(str(save_dir))
    summary_path = save_dir / "summary.json"

    device = torch.device(args_cli.device)
    perturbation_rng = np.random.default_rng(args_cli.seed)
    perturbation_cfg = perturb_spec(args_cli.perturb_level)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.viewer.eye = (3.0, 4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.9)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped
    env.seed(args_cli.seed)
    set_camera(env)

    model, normalizer, model_config, epoch = load_policy(args_cli.checkpoint, device=device)
    n_obs_steps = int(model_config["n_obs_steps"])
    chunk_size = int(model_config["chunk_size"])
    obs_dim = int(model_config["obs_dim"])
    action_dim = int(model_config["action_dim"])
    if n_obs_steps != 2 or chunk_size != 8 or obs_dim != 41 or action_dim != 32:
        raise ValueError(
            f"Unexpected checkpoint config: n_obs_steps={n_obs_steps}, chunk_size={chunk_size}, "
            f"obs_dim={obs_dim}, action_dim={action_dim}"
        )

    success_count = 0
    first_grasp_failed_count = 0
    pure_approach_failed_count = 0
    transport_failed_count = 0
    recovery_candidate_count = 0
    failure_index = 0
    min_left_dist_values: list[float] = []
    object_motion_values: list[float] = []
    summary_records: list[dict[str, Any]] = []

    try:
        for rollout_idx in range(args_cli.num_rollouts):
            obs, _ = env.reset()
            rollout_perturbation = sample_rollout_perturbation(args_cli.perturb_level, perturbation_rng)
            pose_perturb_applied, pose_perturb_warning = apply_object_pose_perturbation(env, rollout_perturbation)
            if pose_perturb_applied:
                obs, _ = refresh_obs_after_perturbation(env)
            else:
                rollout_perturbation["xy_offset"] = np.zeros(2, dtype=np.float32)
                rollout_perturbation["yaw_deg"] = 0.0
                rollout_perturbation["yaw_rad"] = 0.0
                print(f"[WARNING] rollout={rollout_idx} perturbation disabled: {pose_perturb_warning}", flush=True)

            policy_obs = unwrap_policy_obs(obs)
            flat_obs = maybe_add_obs_noise(flatten_policy_obs(policy_obs).to(device), args_cli.obs_noise_std)
            obs_history = flat_obs.unsqueeze(1).repeat(1, n_obs_steps, 1)
            step_records: list[dict[str, Any]] = []
            env_step_idx = 0
            rollout_success = False
            rollout_terminated = False
            rollout_truncated = False
            end_step = None

            while env_step_idx < args_cli.horizon:
                with torch.no_grad():
                    normalized_obs_history = normalizer.normalize_obs(obs_history)
                    action_chunk_norm = model(normalized_obs_history)
                    action_chunk_raw = normalizer.denormalize_action(action_chunk_norm)

                exec_horizon = min(args_cli.exec_horizon, chunk_size, args_cli.horizon - env_step_idx)
                for exec_idx in range(exec_horizon):
                    raw_action = action_chunk_raw[:, exec_idx, :].detach().float().cpu().numpy()[0]
                    env_action = action_chunk_raw[:, exec_idx, :].to(env.device)
                    if args_cli.action_noise_std > 0.0:
                        env_action = env_action + torch.randn_like(env_action) * args_cli.action_noise_std
                    obs, _, terminated, truncated, _ = env.step(env_action)
                    policy_obs = unwrap_policy_obs(obs)
                    flat_obs_now = maybe_add_obs_noise(
                        flatten_policy_obs(policy_obs).to(device),
                        args_cli.obs_noise_std,
                    )
                    step_records.append(
                        build_step_record(
                            env=env,
                            policy_obs=policy_obs,
                            flat_obs=flat_obs_now.detach().float().cpu().numpy()[0],
                            raw_action=raw_action,
                            step_index=env_step_idx,
                        )
                    )
                    obs_history = torch.cat([obs_history[:, 1:, :], flat_obs_now.unsqueeze(1)], dim=1)

                    success = False
                    if success_term is not None:
                        success = tensor_flag_to_bool(success_term.func(env, **success_term.params))
                    rollout_terminated = tensor_flag_to_bool(terminated)
                    rollout_truncated = tensor_flag_to_bool(truncated)
                    if success:
                        rollout_success = True
                        end_step = env_step_idx
                        break
                    if rollout_terminated or rollout_truncated:
                        end_step = env_step_idx
                        break
                    env_step_idx += 1
                if rollout_success or rollout_terminated or rollout_truncated:
                    break

            if end_step is None and step_records:
                end_step = int(step_records[-1]["step_index"])

            if rollout_success:
                success_count += 1
                print(f"[INFO] rollout={rollout_idx} success=True", flush=True)
                continue

            if not step_records:
                print(f"[WARNING] rollout={rollout_idx} failed with no step records", flush=True)
                continue

            metrics = compute_metrics(step_records)
            estimated_phase, failure_reason, recovery_candidate = classify_failure(metrics, rollout_success)
            min_left_dist_values.append(metrics.min_left_dist)
            object_motion_values.append(metrics.object_motion_range)

            if estimated_phase == "first_grasp_failed":
                first_grasp_failed_count += 1
            elif estimated_phase == "pure_approach_failed":
                pure_approach_failed_count += 1
            elif estimated_phase == "transport_failed":
                transport_failed_count += 1
            if recovery_candidate:
                recovery_candidate_count += 1

            window_start = max(0, metrics.min_left_idx - args_cli.window_before)
            window_end = min(len(step_records), metrics.min_left_idx + args_cli.window_after + 1)
            saved_slice = slice(window_start, window_end)
            save_step_indices = metrics.step_indices[saved_slice]

            failure_payload = {
                "failure_id": f"failure_{failure_index:03d}",
                "rollout_idx": int(rollout_idx),
                "success": False,
                "failure_reason": failure_reason,
                "estimated_failure_phase": estimated_phase,
                "checkpoint": args_cli.checkpoint,
                "checkpoint_epoch": epoch,
                "perturb_params": {
                    "level": rollout_perturbation["level"],
                    "xy_offset": [float(x) for x in rollout_perturbation["xy_offset"].tolist()],
                    "yaw_deg": float(rollout_perturbation["yaw_deg"]),
                    "yaw_rad": float(rollout_perturbation["yaw_rad"]),
                    "xy_max": float(perturbation_cfg["xy_max"]),
                    "yaw_deg_max": float(perturbation_cfg["yaw_deg_max"]),
                },
                "step_indices": save_step_indices.tolist(),
                "saved_window_start_step": int(save_step_indices[0]),
                "saved_window_end_step": int(save_step_indices[-1]),
                "flat_obs": metrics.flat_obs[saved_slice],
                "raw_action": metrics.raw_action[saved_slice],
                "object_pos": metrics.object_pos[saved_slice],
                "object_pos_sequence": metrics.object_pos[saved_slice],
                "object_rot": metrics.object_rot[saved_slice],
                "object_rot_sequence": metrics.object_rot[saved_slice],
                "left_eef_pos": metrics.left_eef_pos[saved_slice],
                "left_eef_pos_sequence": metrics.left_eef_pos[saved_slice],
                "right_eef_pos": metrics.right_eef_pos[saved_slice],
                "right_eef_pos_sequence": metrics.right_eef_pos[saved_slice],
                "hand_joint_state": metrics.hand_joint_state[saved_slice],
                "hand_joint_state_sequence": metrics.hand_joint_state[saved_slice],
                "left_object_dist": metrics.left_dist[saved_slice],
                "left_object_dist_sequence": metrics.left_dist[saved_slice],
                "right_object_dist": metrics.right_dist[saved_slice],
                "right_object_dist_sequence": metrics.right_dist[saved_slice],
                "object_motion_range": float(metrics.object_motion_range),
                "time_of_min_left_object_dist": int(metrics.step_indices[metrics.min_left_idx]),
                "min_left_object_dist": float(metrics.min_left_dist),
                "final_left_object_dist": float(metrics.final_left_dist),
                "query_step_suggestion": int(max(0, metrics.step_indices[metrics.min_left_idx] - 8)),
                "terminated": bool(rollout_terminated),
                "truncated": bool(rollout_truncated),
                "end_step": None if end_step is None else int(end_step),
            }
            failure_path = save_dir / f"failure_{failure_index:03d}.pkl"
            save_pickle(failure_path, failure_payload)

            summary_records.append(
                {
                    "failure_id": failure_payload["failure_id"],
                    "rollout_idx": int(rollout_idx),
                    "failure_path": str(failure_path),
                    "failure_reason": failure_reason,
                    "estimated_failure_phase": estimated_phase,
                    "perturb_params": failure_payload["perturb_params"],
                    "time_of_min_left_object_dist": failure_payload["time_of_min_left_object_dist"],
                    "min_left_object_dist": failure_payload["min_left_object_dist"],
                    "final_left_object_dist": failure_payload["final_left_object_dist"],
                    "object_motion_range": failure_payload["object_motion_range"],
                    "recovery_candidate": recovery_candidate,
                }
            )
            print(
                f"[INFO] rollout={rollout_idx} phase={estimated_phase} "
                f"left_min={metrics.min_left_dist:.4f} motion={metrics.object_motion_range:.4f} "
                f"saved={failure_path}",
                flush=True,
            )
            failure_index += 1

        total_failures = args_cli.num_rollouts - success_count
        summary_payload = {
            "total_rollouts": int(args_cli.num_rollouts),
            "success_count": int(success_count),
            "total_failures": int(total_failures),
            "first_grasp_failed_count": int(first_grasp_failed_count),
            "pure_approach_failed_count": int(pure_approach_failed_count),
            "transport_failed_count": int(transport_failed_count),
            "recovery_candidate_count": int(recovery_candidate_count),
            "avg_min_left_object_dist": float(np.mean(min_left_dist_values)) if min_left_dist_values else None,
            "avg_object_motion_range": float(np.mean(object_motion_values)) if object_motion_values else None,
            "checkpoint": args_cli.checkpoint,
            "perturb_level": args_cli.perturb_level,
            "exec_horizon": int(args_cli.exec_horizon),
            "records": summary_records,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"[INFO] summary_path={summary_path}", flush=True)
        print(f"[INFO] first_grasp_failed_count={first_grasp_failed_count}", flush=True)
    finally:
        print("[INFO] Skipping env.close(); relying on simulation_app.close()", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
