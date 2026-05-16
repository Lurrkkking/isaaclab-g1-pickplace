#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pathlib
import pickle
import random
import sys
import traceback
import types
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Optional

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

import gymnasium as gym

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

FAILURE_RULES = [
    "approach_failed: left/right hand never get close to object; min_left_object_dist stays large.",
    "left_grasp_failed: left hand gets near object once, but object motion remains small and handoff never emerges.",
    "handoff_failed: object becomes close to both hands or enters right-hand receive zone, then right-object distance re-opens and rollout fails.",
    "right_grasp_failed: right hand reaches object, but object does not track right-hand motion or quickly drops/drifts away.",
    "transport_failed: object clearly moves, but target is not reached after pickup/transfer.",
    "release_failed: object reaches target vicinity but success is not completed.",
    "timeout_near_success: horizon expires while object remains near target or close to a likely successful final state.",
    "unknown_failed: fallback bucket when no heuristic is decisive.",
]


parser = argparse.ArgumentParser(description="Collect ACT near-miss failures for G1 locomanipulation.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/root/autodl-tmp/act_g1/outputs/g1_act_100ep/checkpoints/epoch_0500.pt",
)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--horizon", type=int, default=120)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--tail_k", type=int, default=80)
parser.add_argument(
    "--perturb_levels",
    type=str,
    nargs="+",
    default=["medium", "hard"],
    choices=["none", "mild", "medium", "hard"],
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/root/autodl-tmp/IsaacLab/logs/nearmiss_failures",
)
parser.add_argument("--verbose_steps", action="store_true", default=False)

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
            raise KeyError(f"Missing observation key '{key}'")
        value = policy_obs[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        flat_parts.append(value)
    flat_obs = torch.cat(flat_parts, dim=-1)
    if flat_obs.shape[-1] != 41:
        raise ValueError(f"Expected flat obs dim 41, got {tuple(flat_obs.shape)}")
    return flat_obs


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


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    ckpt = pathlib.Path(checkpoint_path)
    if ckpt.is_file():
        return str(ckpt)

    fallback_candidates = [
        "/root/autodl-tmp/act_g1/outputs/g1_act_100ep/checkpoints/epoch_0500.pt",
        "/root/autodl-tmp/act_g1/outputs/g1_act_100ep/checkpoints/latest.pt",
        "/root/autodl-tmp/act_g1/outputs/g1_act_50ep/checkpoints/latest.pt",
    ]
    for candidate in fallback_candidates:
        if pathlib.Path(candidate).is_file():
            print(
                f"[WARNING] checkpoint not found at {checkpoint_path}; falling back to {candidate}",
                flush=True,
            )
            return candidate
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}. "
        f"Tried fallbacks: {fallback_candidates}"
    )


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
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    model_cls = import_act_model_class()
    model = model_cls(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    normalizer = ACTNormalizer(stats=checkpoint["normalization_stats"], device=device)
    return model, normalizer, checkpoint["model_config"], checkpoint.get("epoch"), checkpoint_path


def to_numpy_1d(value, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=dtype)
    if value.ndim > 1:
        value = value[0]
    return value.astype(dtype, copy=False)


def tensor_to_list(value) -> Optional[list[float]]:
    if value is None:
        return None
    return to_numpy_1d(value).tolist()


def get_object_pos(policy_obs) -> np.ndarray:
    if "object_pos" in policy_obs:
        return to_numpy_1d(policy_obs["object_pos"])
    if "object" in policy_obs:
        return to_numpy_1d(policy_obs["object"])[:3]
    raise KeyError("Missing object/object_pos in policy observation.")


def get_object_rot(env) -> Optional[list[float]]:
    try:
        scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
        rigid_object = scene["object"]
        root_state_w = rigid_object.data.root_state_w
        return tensor_to_list(root_state_w[0, 3:7])
    except Exception:
        return None


def maybe_get_object_target(success_term, object_pos: np.ndarray) -> Optional[np.ndarray]:
    if success_term is None:
        return None
    params = getattr(success_term, "params", None)
    if not isinstance(params, dict):
        return None
    required_keys = {"min_x", "max_x", "min_y", "max_y"}
    if not required_keys.issubset(params.keys()):
        return None
    target_x = 0.5 * (float(params["min_x"]) + float(params["max_x"]))
    target_y = 0.5 * (float(params["min_y"]) + float(params["max_y"]))
    target_z = float(object_pos[2])
    if "max_height" in params:
        target_z = min(target_z, float(params["max_height"]))
    return np.asarray([target_x, target_y, target_z], dtype=np.float32)


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
class FailureClassification:
    phase: str
    handoff_candidate: bool
    min_left_object_dist: float
    min_right_object_dist: float
    final_left_object_dist: float
    final_right_object_dist: float
    min_object_target_dist: Optional[float]
    final_object_target_dist: Optional[float]
    object_motion_range: float
    query_step_index: int
    rule_notes: list[str]


def _movement_range(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    center = points.mean(axis=0)
    return float(np.linalg.norm(points - center[None, :], axis=1).max())


def _safe_min(values: list[float]) -> float:
    return float(min(values)) if values else float("inf")


def classify_failure_phase(
    step_records: list[dict[str, Any]],
    failure_reason: str,
) -> FailureClassification:
    if not step_records:
        return FailureClassification(
            phase="unknown_failed",
            handoff_candidate=False,
            min_left_object_dist=float("inf"),
            min_right_object_dist=float("inf"),
            final_left_object_dist=float("inf"),
            final_right_object_dist=float("inf"),
            min_object_target_dist=None,
            final_object_target_dist=None,
            object_motion_range=0.0,
            query_step_index=-1,
            rule_notes=["empty trajectory; no post-step records were captured"],
        )
    object_positions = np.asarray([r["object_pos"] for r in step_records], dtype=np.float32)
    left_positions = np.asarray([r["left_eef_pos"] for r in step_records], dtype=np.float32)
    right_positions = np.asarray([r["right_eef_pos"] for r in step_records], dtype=np.float32)
    left_dist = np.linalg.norm(object_positions - left_positions, axis=1)
    right_dist = np.linalg.norm(object_positions - right_positions, axis=1)
    target_list = [r["target_pos"] for r in step_records if r["target_pos"] is not None]
    target_dist = None
    if target_list:
        target = np.asarray(target_list[0], dtype=np.float32)
        target_dist = np.linalg.norm(object_positions - target[None, :], axis=1)

    min_left = float(left_dist.min())
    min_right = float(right_dist.min())
    final_left = float(left_dist[-1])
    final_right = float(right_dist[-1])
    min_target = float(target_dist.min()) if target_dist is not None else None
    final_target = float(target_dist[-1]) if target_dist is not None else None
    object_motion_range = _movement_range(object_positions)

    near_left = left_dist < 0.085
    near_right = right_dist < 0.09
    dual_near = np.logical_and(left_dist < 0.10, right_dist < 0.12)
    handoff_candidate = bool(np.any(dual_near) or np.any(np.logical_and(near_left, right_dist < 0.14)))

    min_right_idx = int(np.argmin(right_dist))
    min_left_idx = int(np.argmin(left_dist))
    right_reopened = bool(min_right < 0.09 and np.any(right_dist[min_right_idx:] > min_right + 0.05))
    left_contact_before_right = bool(min_left_idx <= min_right_idx and min_left < 0.10)

    right_suffix_move = 0.0
    object_suffix_move = 0.0
    if min_right_idx < len(step_records):
        right_suffix_move = float(np.linalg.norm(right_positions[min_right_idx:] - right_positions[min_right_idx], axis=1).max())
        object_suffix_move = float(np.linalg.norm(object_positions[min_right_idx:] - object_positions[min_right_idx], axis=1).max())
    object_not_following_right = bool(min_right < 0.08 and right_suffix_move > 0.06 and object_suffix_move < 0.035)
    object_drop_after_right = bool(min_right < 0.08 and final_right > 0.12 and object_suffix_move < 0.06)

    timeout_near_success = bool(
        failure_reason == "timeout"
        and (
            (min_target is not None and min_target < 0.10)
            or (final_target is not None and final_target < 0.12)
            or (min_right < 0.06 and handoff_candidate)
        )
    )
    release_failed = bool(min_target is not None and min_target < 0.08 and final_target is not None and final_target < 0.14)
    handoff_failed = bool(handoff_candidate and left_contact_before_right and right_reopened and final_right > 0.09)
    right_grasp_failed = bool(not handoff_failed and (object_not_following_right or object_drop_after_right))
    approach_failed = bool(min_left > 0.14 and min_right > 0.14)
    left_grasp_failed = bool(min_left < 0.09 and not handoff_candidate and object_motion_range < 0.04)
    transport_failed = bool(
        object_motion_range > 0.08
        and not release_failed
        and not right_grasp_failed
        and not handoff_failed
        and (min_target is None or min_target > 0.12)
    )

    phase = "unknown_failed"
    rule_notes: list[str] = []
    if timeout_near_success:
        phase = "timeout_near_success"
        rule_notes.append("timeout with near-success geometry")
    elif release_failed:
        phase = "release_failed"
        rule_notes.append("object reached target vicinity but task not completed")
    elif handoff_failed:
        phase = "handoff_failed"
        rule_notes.append("dual-hand receive geometry appeared, then right-hand contact reopened")
    elif right_grasp_failed:
        phase = "right_grasp_failed"
        rule_notes.append("right hand approached object but object did not follow right-hand motion")
    elif approach_failed:
        phase = "approach_failed"
        rule_notes.append("both end-effectors stayed far from the object")
    elif left_grasp_failed:
        phase = "left_grasp_failed"
        rule_notes.append("left hand approached object but pickup/transfer never started")
    elif transport_failed:
        phase = "transport_failed"
        rule_notes.append("object moved substantially but target was not reached")
    else:
        rule_notes.append("no heuristic was decisive")

    if phase in {"handoff_failed", "right_grasp_failed"}:
        query_idx = min_right_idx
    elif min_target is not None:
        query_idx = int(np.argmin(target_dist))
    else:
        query_idx = len(step_records) - 1

    return FailureClassification(
        phase=phase,
        handoff_candidate=handoff_candidate,
        min_left_object_dist=min_left,
        min_right_object_dist=min_right,
        final_left_object_dist=final_left,
        final_right_object_dist=final_right,
        min_object_target_dist=min_target,
        final_object_target_dist=final_target,
        object_motion_range=object_motion_range,
        query_step_index=int(step_records[query_idx]["step_index"]),
        rule_notes=rule_notes,
    )


def build_step_record(
    env,
    policy_obs,
    flat_obs: np.ndarray,
    raw_action: np.ndarray,
    env_action: np.ndarray,
    step_index: int,
    success: bool,
    terminated: bool,
    truncated: bool,
    perturbation: dict[str, Any],
    target_pos: Optional[np.ndarray],
) -> dict[str, Any]:
    return {
        "flat_obs": flat_obs.astype(np.float32, copy=True),
        "raw_action": raw_action.astype(np.float32, copy=True),
        "env_action": env_action.astype(np.float32, copy=True),
        "step_index": int(step_index),
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "object_pos": get_object_pos(policy_obs).astype(np.float32, copy=True).tolist(),
        "object_rot": get_object_rot(env),
        "left_eef_pos": tensor_to_list(policy_obs.get("left_eef_pos")),
        "right_eef_pos": tensor_to_list(policy_obs.get("right_eef_pos")),
        "hand_joint_state": tensor_to_list(policy_obs.get("hand_joint_state")),
        "target_pos": target_pos.tolist() if target_pos is not None else None,
        "perturb_params": {
            "level": perturbation["level"],
            "xy_offset": [float(x) for x in perturbation["xy_offset"].tolist()],
            "yaw_deg": float(perturbation["yaw_deg"]),
            "yaw_rad": float(perturbation["yaw_rad"]),
        },
    }


def save_pickle(path: pathlib.Path, payload: Any) -> None:
    ensure_dir(str(path.parent))
    with path.open("wb") as f:
        pickle.dump(payload, f)


def set_camera(env) -> None:
    env.unwrapped.sim.set_camera_view(eye=[3.0, 4.0, 2.2], target=[0.0, 0.2, 0.9])


def maybe_render_mode() -> Optional[str]:
    return None


def main() -> None:
    if args_cli.exec_horizon < 1:
        raise ValueError("exec_horizon must be >= 1")
    if args_cli.tail_k < 1:
        raise ValueError("tail_k must be >= 1")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    output_dir = pathlib.Path(args_cli.output_dir)
    ensure_dir(str(output_dir))
    summary_path = output_dir / "summary.json"

    device = torch.device(args_cli.device)
    perturbation_rng = np.random.default_rng(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.viewer.eye = (3.0, 4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.9)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=maybe_render_mode()).unwrapped
    env.seed(args_cli.seed)
    set_camera(env)

    model, normalizer, model_config, epoch, resolved_checkpoint = load_policy(args_cli.checkpoint, device=device)
    n_obs_steps = int(model_config["n_obs_steps"])
    chunk_size = int(model_config["chunk_size"])
    obs_dim = int(model_config["obs_dim"])
    action_dim = int(model_config["action_dim"])
    if n_obs_steps != 2 or chunk_size != 8 or obs_dim != 41 or action_dim != 32:
        raise ValueError(
            f"Unexpected checkpoint config: n_obs_steps={n_obs_steps}, chunk_size={chunk_size}, "
            f"obs_dim={obs_dim}, action_dim={action_dim}"
        )

    print("[INFO] Failure phase rules:", flush=True)
    for rule in FAILURE_RULES:
        print(f"[INFO]   - {rule}", flush=True)
    print(
        f"[INFO] checkpoint={resolved_checkpoint} epoch={epoch} "
        f"num_rollouts={args_cli.num_rollouts} horizon={args_cli.horizon} exec_horizon={args_cli.exec_horizon} "
        f"perturb_levels={args_cli.perturb_levels} tail_k={args_cli.tail_k}",
        flush=True,
    )

    summary_records = []
    phase_counter: Counter[str] = Counter()
    success_count = 0
    failure_count = 0
    recovery_candidate_count = 0
    handoff_failed_count = 0
    right_grasp_failed_count = 0
    failure_index = 0

    try:
        for rollout_idx in range(args_cli.num_rollouts):
            perturb_level = args_cli.perturb_levels[rollout_idx % len(args_cli.perturb_levels)]
            perturb_cfg = perturb_spec(perturb_level)
            obs, _ = env.reset()
            rollout_perturbation = sample_rollout_perturbation(perturb_level, perturbation_rng)
            pose_perturb_applied, pose_perturb_warning = apply_object_pose_perturbation(env, rollout_perturbation)
            if pose_perturb_applied:
                obs, _ = refresh_obs_after_perturbation(env)
            else:
                rollout_perturbation["xy_offset"] = np.zeros(2, dtype=np.float32)
                rollout_perturbation["yaw_deg"] = 0.0
                rollout_perturbation["yaw_rad"] = 0.0
                print(f"[WARNING] rollout={rollout_idx} perturbation disabled: {pose_perturb_warning}", flush=True)

            policy_obs = unwrap_policy_obs(obs)
            flat_obs = flatten_policy_obs(policy_obs).to(device)
            obs_history = flat_obs.unsqueeze(1).repeat(1, n_obs_steps, 1)
            step_records: list[dict[str, Any]] = []
            failure_tail: deque[dict[str, Any]] = deque(maxlen=args_cli.tail_k)
            rollout_success = False
            rollout_terminated = False
            rollout_truncated = False
            success_step = None
            end_step = None
            env_step_idx = 0
            last_policy_obs = policy_obs

            while env_step_idx < args_cli.horizon:
                with torch.no_grad():
                    normalized_obs_history = normalizer.normalize_obs(obs_history)
                    action_chunk_norm = model(normalized_obs_history)
                    action_chunk_raw = normalizer.denormalize_action(action_chunk_norm)

                exec_horizon = min(args_cli.exec_horizon, chunk_size, args_cli.horizon - env_step_idx)
                for exec_idx in range(exec_horizon):
                    raw_action = action_chunk_raw[:, exec_idx, :].detach().float().cpu().numpy()[0]
                    env_action = action_chunk_raw[:, exec_idx, :].to(env.device)
                    obs, _, terminated, truncated, _ = env.step(env_action)
                    policy_obs = unwrap_policy_obs(obs)
                    flat_obs_now = flatten_policy_obs(policy_obs).detach().float().cpu().numpy()[0]
                    last_policy_obs = policy_obs
                    object_pos = get_object_pos(policy_obs)
                    target_pos = maybe_get_object_target(success_term, object_pos)
                    success = False
                    if success_term is not None:
                        success = tensor_flag_to_bool(success_term.func(env, **success_term.params))
                    terminated_flag = tensor_flag_to_bool(terminated)
                    truncated_flag = tensor_flag_to_bool(truncated)
                    record = build_step_record(
                        env=env,
                        policy_obs=policy_obs,
                        flat_obs=flat_obs_now,
                        raw_action=raw_action,
                        env_action=env_action.detach().float().cpu().numpy()[0],
                        step_index=env_step_idx,
                        success=success,
                        terminated=terminated_flag,
                        truncated=truncated_flag,
                        perturbation=rollout_perturbation,
                        target_pos=target_pos,
                    )
                    step_records.append(record)
                    failure_tail.append(record)
                    obs_history = torch.cat(
                        [obs_history[:, 1:, :], torch.from_numpy(flat_obs_now).to(device).unsqueeze(0).unsqueeze(1)],
                        dim=1,
                    )

                    if args_cli.verbose_steps:
                        left_dist = float(np.linalg.norm(np.asarray(record["object_pos"]) - np.asarray(record["left_eef_pos"])))
                        right_dist = float(np.linalg.norm(np.asarray(record["object_pos"]) - np.asarray(record["right_eef_pos"])))
                        print(
                            f"[STEP] rollout={rollout_idx} step={env_step_idx} perturb={perturb_level} "
                            f"success={success} terminated={terminated_flag} truncated={truncated_flag} "
                            f"left_obj={left_dist:.4f} right_obj={right_dist:.4f}",
                            flush=True,
                        )

                    if success:
                        rollout_success = True
                        success_step = env_step_idx
                        end_step = env_step_idx
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        break
                    if terminated_flag or truncated_flag:
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        end_step = env_step_idx
                        break

                    env_step_idx += 1

                if rollout_success or rollout_terminated or rollout_truncated:
                    break

            if end_step is None:
                end_step = step_records[-1]["step_index"] if step_records else None

            failure_reason = infer_failure_reason(
                success=rollout_success,
                terminated=rollout_terminated,
                truncated=rollout_truncated,
                end_step=end_step,
                horizon=args_cli.horizon,
            )
            if not pose_perturb_applied and pose_perturb_warning is not None:
                failure_reason = f"{failure_reason}; object_pose_perturbation_disabled"

            if rollout_success:
                success_count += 1
                print(
                    f"[INFO] rollout={rollout_idx} success=True success_step={success_step} perturb={perturb_level}",
                    flush=True,
                )
                continue

            failure_count += 1
            classification = classify_failure_phase(step_records=step_records, failure_reason=failure_reason.split(";")[0])
            phase_counter[classification.phase] += 1
            if classification.handoff_candidate:
                recovery_candidate_count += 1
            if classification.phase == "handoff_failed":
                handoff_failed_count += 1
            if classification.phase == "right_grasp_failed":
                right_grasp_failed_count += 1

            tail_list = list(failure_tail)
            if tail_list and classification.query_step_index >= 0:
                query_tail_index = max(0, classification.query_step_index - int(tail_list[0]["step_index"]))
                query_tail_index = int(min(query_tail_index, len(tail_list) - 1))
            else:
                query_tail_index = max(0, len(tail_list) - 1)
            failure_path = output_dir / f"failure_{failure_index:03d}.pkl"
            failure_payload = {
                "failure_id": f"failure_{failure_index:03d}",
                "rollout_index": rollout_idx,
                "checkpoint": resolved_checkpoint,
                "checkpoint_epoch": epoch,
                "task": args_cli.task,
                "perturb_params": {
                    "level": rollout_perturbation["level"],
                    "xy_offset": [float(x) for x in rollout_perturbation["xy_offset"].tolist()],
                    "yaw_deg": float(rollout_perturbation["yaw_deg"]),
                    "yaw_rad": float(rollout_perturbation["yaw_rad"]),
                    "xy_max": float(perturb_cfg["xy_max"]),
                    "yaw_deg_max": float(perturb_cfg["yaw_deg_max"]),
                },
                "tail_k": args_cli.tail_k,
                "step_count": len(step_records),
                "end_step": end_step,
                "success": False,
                "terminated": rollout_terminated,
                "truncated": rollout_truncated,
                "failure_reason": failure_reason,
                "failure_phase": classification.phase,
                "estimated_failure_phase": classification.phase,
                "handoff_candidate": classification.handoff_candidate,
                "min_left_object_dist": classification.min_left_object_dist,
                "min_right_object_dist": classification.min_right_object_dist,
                "final_left_object_dist": classification.final_left_object_dist,
                "final_right_object_dist": classification.final_right_object_dist,
                "min_object_target_dist": classification.min_object_target_dist,
                "final_object_target_dist": classification.final_object_target_dist,
                "object_motion_range": classification.object_motion_range,
                "query_step_index": classification.query_step_index,
                "query_tail_index": query_tail_index,
                "rule_notes": classification.rule_notes,
                "failure_tail": tail_list,
            }
            save_pickle(failure_path, failure_payload)

            summary_record = {
                "failure_id": failure_payload["failure_id"],
                "failure_path": str(failure_path),
                "rollout_index": rollout_idx,
                "success": False,
                "terminated": rollout_terminated,
                "truncated": rollout_truncated,
                "failure_reason": failure_reason,
                "failure_phase": classification.phase,
                "estimated_failure_phase": classification.phase,
                "handoff_candidate": classification.handoff_candidate,
                "min_left_object_dist": classification.min_left_object_dist,
                "min_right_object_dist": classification.min_right_object_dist,
                "final_left_object_dist": classification.final_left_object_dist,
                "final_right_object_dist": classification.final_right_object_dist,
                "min_object_target_dist": classification.min_object_target_dist,
                "object_motion_range": classification.object_motion_range,
                "query_step_index": classification.query_step_index,
                "perturb_params": failure_payload["perturb_params"],
                "rule_notes": classification.rule_notes,
            }
            summary_records.append(summary_record)
            print(
                f"[INFO] rollout={rollout_idx} success=False phase={classification.phase} "
                f"handoff_candidate={classification.handoff_candidate} "
                f"left_min={classification.min_left_object_dist:.4f} "
                f"right_min={classification.min_right_object_dist:.4f} "
                f"target_min={classification.min_object_target_dist} "
                f"failure_path={failure_path}",
                flush=True,
            )
            failure_index += 1

        summary_payload = {
            "total_rollouts": args_cli.num_rollouts,
            "total_failures": failure_count,
            "success_count": success_count,
            "failure_phase_counts": dict(sorted(phase_counter.items())),
            "handoff_failed_count": handoff_failed_count,
            "right_grasp_failed_count": right_grasp_failed_count,
            "recovery_candidate_count": recovery_candidate_count,
            "tail_k": args_cli.tail_k,
            "horizon": args_cli.horizon,
            "exec_horizon": args_cli.exec_horizon,
            "checkpoint": resolved_checkpoint,
            "perturb_levels": list(args_cli.perturb_levels),
            "failure_phase_rules": FAILURE_RULES,
            "records": summary_records,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        print(f"[INFO] summary_path={summary_path}", flush=True)
        print(f"[INFO] total_rollouts={args_cli.num_rollouts}", flush=True)
        print(f"[INFO] total_failures={failure_count}", flush=True)
        print(f"[INFO] success_count={success_count}", flush=True)
        print(f"[INFO] failure_phase_counts={dict(sorted(phase_counter.items()))}", flush=True)
        print(f"[INFO] handoff_failed_count={handoff_failed_count}", flush=True)
        print(f"[INFO] right_grasp_failed_count={right_grasp_failed_count}", flush=True)
        print(f"[INFO] recovery_candidate_count={recovery_candidate_count}", flush=True)
    finally:
        print("[INFO] skipping env.close(); relying on simulation_app.close()", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
