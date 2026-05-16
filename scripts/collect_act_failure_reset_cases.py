#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import random
import sys
import traceback
import types
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


parser = argparse.ArgumentParser(description="Collect ACT failure-targeted reset cases for G1 locomanipulation.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_rollouts", type=int, default=100)
parser.add_argument("--horizon", type=int, default=400)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--exec_horizon", type=int, default=4)
parser.add_argument("--perturb_level", type=str, choices=["none", "mild", "medium", "hard"], default="hard")
parser.add_argument("--object_xy_range", type=float, default=None)
parser.add_argument("--object_yaw_range_deg", type=float, default=None)
parser.add_argument("--obs_noise_std", type=float, default=0.0)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, required=True)

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
        elif value.ndim != 2:
            raise ValueError(f"Unexpected shape for obs key '{key}': {tuple(value.shape)}")
        flat_parts.append(value)

    flat_obs = torch.cat(flat_parts, dim=-1)
    if flat_obs.shape[-1] != 41:
        raise ValueError(f"Flat obs last dim should be 41, got {tuple(flat_obs.shape)}")
    return flat_obs


def set_camera(env) -> None:
    env.unwrapped.sim.set_camera_view(eye=[3.0, 4.0, 2.2], target=[0.0, 0.2, 0.9])


def perturb_spec(level: str) -> dict[str, float]:
    if level not in PERTURB_CONFIGS:
        raise ValueError(f"Unsupported perturb_level: {level}")
    return dict(PERTURB_CONFIGS[level])


def resolve_perturb_spec(level: str, object_xy_range: Optional[float], object_yaw_range_deg: Optional[float]) -> dict[str, float]:
    spec = perturb_spec(level)
    if object_xy_range is not None:
        spec["xy_max"] = float(object_xy_range)
    if object_yaw_range_deg is not None:
        spec["yaw_deg_max"] = float(object_yaw_range_deg)
    return spec


def sample_rollout_perturbation(level: str, rng: np.random.Generator, spec: Optional[dict[str, float]] = None) -> dict[str, Any]:
    if spec is None:
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


def get_rigid_object(env):
    scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
    return scene["object"], scene


def get_object_root_state(env) -> torch.Tensor:
    rigid_object, _ = get_rigid_object(env)
    return rigid_object.data.root_state_w[0].detach().clone()


def get_object_initial_pose(env) -> tuple[list[float], list[float]]:
    root_state_w = get_object_root_state(env)
    pos = root_state_w[:3].detach().float().cpu().tolist()
    quat = root_state_w[3:7].detach().float().cpu().tolist()
    return [float(x) for x in pos], [float(x) for x in quat]


def apply_object_pose_perturbation(env, perturbation: dict[str, Any]) -> tuple[bool, Optional[str]]:
    if perturbation["level"] == "none":
        return True, None

    xy_offset = perturbation["xy_offset"]
    yaw_rad = float(perturbation["yaw_rad"])
    if float(np.linalg.norm(xy_offset)) == 0.0 and yaw_rad == 0.0:
        return True, None

    try:
        rigid_object, scene = get_rigid_object(env)
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
    ckpt_path = pathlib.Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = load_torch_checkpoint(str(ckpt_path), map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        raise KeyError("Checkpoint is missing normalization_stats")

    model_cls = import_act_model_class()
    model = model_cls(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalizer = ACTNormalizer(stats=stats, device=device)
    return model, normalizer, {"model_config": model_config, "epoch": checkpoint.get("epoch")}


def to_numpy_1d(value, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=dtype)
    if value.ndim > 1:
        value = value[0]
    return value.astype(dtype, copy=False)


def get_object_pos(policy_obs, env) -> np.ndarray:
    if "object_pos" in policy_obs:
        return to_numpy_1d(policy_obs["object_pos"])
    if "object" in policy_obs:
        return to_numpy_1d(policy_obs["object"])[:3]
    return get_object_root_state(env)[:3].detach().float().cpu().numpy()


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


def compute_object_motion_range(object_positions: list[np.ndarray]) -> float:
    if len(object_positions) <= 1:
        return 0.0
    object_pos = np.asarray(object_positions, dtype=np.float32)
    disp = np.linalg.norm(object_pos - object_pos[0:1], axis=1)
    return float(disp.max())


def summarize_scalar(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarize_xy_offsets(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "x": {"min": None, "max": None, "mean": None},
            "y": {"min": None, "max": None, "mean": None},
            "norm": {"min": None, "max": None, "mean": None},
        }
    offsets = np.asarray([record["object_xy_offset"] for record in records], dtype=np.float64)
    norms = np.linalg.norm(offsets, axis=1)
    return {
        "x": summarize_scalar(offsets[:, 0].tolist()),
        "y": summarize_scalar(offsets[:, 1].tolist()),
        "norm": summarize_scalar(norms.tolist()),
    }


def write_json(path: pathlib.Path, payload: Any) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    if args_cli.exec_horizon < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {args_cli.exec_horizon}")
    if args_cli.object_xy_range is not None and args_cli.object_xy_range < 0.0:
        raise ValueError(f"object_xy_range must be >= 0, got {args_cli.object_xy_range}")
    if args_cli.object_yaw_range_deg is not None and args_cli.object_yaw_range_deg < 0.0:
        raise ValueError(f"object_yaw_range_deg must be >= 0, got {args_cli.object_yaw_range_deg}")
    if args_cli.obs_noise_std < 0.0:
        raise ValueError(f"obs_noise_std must be >= 0, got {args_cli.obs_noise_std}")
    if args_cli.action_noise_std < 0.0:
        raise ValueError(f"action_noise_std must be >= 0, got {args_cli.action_noise_std}")

    save_dir = pathlib.Path(args_cli.save_dir)
    ensure_dir(str(save_dir))
    all_cases_path = save_dir / "all_cases.json"
    failure_cases_path = save_dir / "failure_cases.json"
    summary_path = save_dir / "summary.json"

    device = torch.device(args_cli.device)
    perturbation_rng = np.random.default_rng(args_cli.seed)
    perturbation_cfg = resolve_perturb_spec(
        args_cli.perturb_level,
        object_xy_range=args_cli.object_xy_range,
        object_yaw_range_deg=args_cli.object_yaw_range_deg,
    )

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

    model, normalizer, ckpt_meta = load_policy(args_cli.checkpoint, device=device)
    model_config = ckpt_meta["model_config"]
    n_obs_steps = int(model_config["n_obs_steps"])
    chunk_size = int(model_config["chunk_size"])
    action_dim = int(model_config["action_dim"])
    obs_dim = int(model_config["obs_dim"])
    if n_obs_steps != 2:
        raise ValueError(f"ACT rollout expects n_obs_steps=2, got {n_obs_steps}")
    if chunk_size != 8:
        raise ValueError(f"ACT rollout expects chunk_size=8, got {chunk_size}")
    if obs_dim != 41:
        raise ValueError(f"ACT rollout expects obs_dim=41, got {obs_dim}")
    if action_dim != 32:
        raise ValueError(f"ACT rollout expects action_dim=32, got {action_dim}")

    all_cases: list[dict[str, Any]] = []
    failure_cases: list[dict[str, Any]] = []
    pose_perturb_warned = False

    try:
        for rollout_idx in range(args_cli.num_rollouts):
            obs, _ = env.reset()
            rollout_seed = int(args_cli.seed + rollout_idx)
            rollout_perturbation = sample_rollout_perturbation(
                args_cli.perturb_level,
                perturbation_rng,
                spec=perturbation_cfg,
            )
            pose_perturb_applied = True
            pose_perturb_warning = None
            if args_cli.perturb_level != "none":
                pose_perturb_applied, pose_perturb_warning = apply_object_pose_perturbation(env, rollout_perturbation)
                if pose_perturb_applied:
                    obs, _ = refresh_obs_after_perturbation(env)
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

            object_initial_pos, object_initial_quat = get_object_initial_pose(env)

            policy_obs = unwrap_policy_obs(obs)
            flat_obs = maybe_add_obs_noise(flatten_policy_obs(policy_obs).to(device), args_cli.obs_noise_std)
            obs_history = flat_obs.unsqueeze(1).repeat(1, n_obs_steps, 1)

            success_step = None
            end_step = None
            rollout_success = False
            rollout_terminated = False
            rollout_truncated = False
            env_step_idx = 0
            last_step_idx = None

            left_object_distances: list[float] = []
            object_positions: list[np.ndarray] = []

            while env_step_idx < args_cli.horizon:
                with torch.no_grad():
                    normalized_obs_history = normalizer.normalize_obs(obs_history)
                    action_chunk_norm = model(normalized_obs_history)
                    action_chunk_raw = normalizer.denormalize_action(action_chunk_norm)

                if action_chunk_norm.shape != (1, chunk_size, action_dim):
                    raise ValueError(
                        f"Expected action_chunk shape (1, {chunk_size}, {action_dim}), got {tuple(action_chunk_norm.shape)}"
                    )

                exec_horizon = min(args_cli.exec_horizon, chunk_size, args_cli.horizon - env_step_idx)
                for exec_idx in range(exec_horizon):
                    env_action = action_chunk_raw[:, exec_idx, :].to(env.device)
                    if args_cli.action_noise_std > 0.0:
                        env_action = env_action + torch.randn_like(env_action) * args_cli.action_noise_std
                    if env_action.ndim != 2 or env_action.shape[1] != 32:
                        raise ValueError(f"env.step action must be (num_envs, 32), got {tuple(env_action.shape)}")

                    obs, _, terminated, truncated, _ = env.step(env_action)
                    policy_obs = unwrap_policy_obs(obs)
                    object_pos = get_object_pos(policy_obs, env)
                    left_eef_pos = to_numpy_1d(policy_obs["left_eef_pos"])
                    left_object_distances.append(float(np.linalg.norm(object_pos - left_eef_pos)))
                    object_positions.append(object_pos.astype(np.float32, copy=True))

                    flat_obs = maybe_add_obs_noise(flatten_policy_obs(policy_obs).to(device), args_cli.obs_noise_std)
                    obs_history = torch.cat([obs_history[:, 1:, :], flat_obs.unsqueeze(1)], dim=1)
                    last_step_idx = env_step_idx

                    success = False
                    if success_term is not None:
                        success = tensor_flag_to_bool(success_term.func(env, **success_term.params))
                    terminated_flag = tensor_flag_to_bool(terminated)
                    truncated_flag = tensor_flag_to_bool(truncated)
                    done = terminated_flag or truncated_flag

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

            failure_reason = infer_failure_reason(
                success=rollout_success,
                terminated=rollout_terminated,
                truncated=rollout_truncated,
                end_step=end_step,
                horizon=args_cli.horizon,
            )
            if not pose_perturb_applied and pose_perturb_warning is not None:
                failure_reason = f"{failure_reason}; object_pose_perturbation_disabled"

            min_left_object_dist = float(min(left_object_distances)) if left_object_distances else None
            object_motion_range = compute_object_motion_range(object_positions)

            case_record = {
                "rollout_idx": int(rollout_idx),
                "seed": rollout_seed,
                "perturb_level": args_cli.perturb_level,
                "object_xy_range": float(perturbation_cfg["xy_max"]),
                "object_yaw_range_deg": float(perturbation_cfg["yaw_deg_max"]),
                "object_xy_offset": [float(x) for x in rollout_perturbation["xy_offset"].tolist()],
                "object_yaw_offset_deg": float(rollout_perturbation["yaw_deg"]),
                "object_initial_pos": object_initial_pos,
                "object_initial_quat": object_initial_quat,
                "success": bool(rollout_success),
                "success_step": None if success_step is None else int(success_step),
                "end_step": None if end_step is None else int(end_step),
                "failure_reason": failure_reason,
                "min_left_object_dist": min_left_object_dist,
                "object_motion_range": float(object_motion_range),
            }
            all_cases.append(case_record)
            if not rollout_success:
                failure_cases.append(case_record)

            print(
                f"rollout {rollout_idx} summary: success={rollout_success}, "
                f"success_step={success_step}, end_step={end_step}, "
                f"perturb_level={args_cli.perturb_level}, "
                f"object_xy_range={perturbation_cfg['xy_max']:.3f}, "
                f"object_yaw_range_deg={perturbation_cfg['yaw_deg_max']:.1f}, "
                f"object_xy_offset={[round(x, 4) for x in case_record['object_xy_offset']]}, "
                f"object_yaw_offset_deg={case_record['object_yaw_offset_deg']:.3f}, "
                f"failure_reason={failure_reason}, "
                f"min_left_object_dist={min_left_object_dist}, "
                f"object_motion_range={object_motion_range:.6f}",
                flush=True,
            )

        success_count = sum(1 for record in all_cases if record["success"])
        failure_count = len(all_cases) - success_count
        summary_payload = {
            "total_rollouts": len(all_cases),
            "success_count": success_count,
            "failure_count": failure_count,
            "perturb_level": args_cli.perturb_level,
            "object_xy_range": float(perturbation_cfg["xy_max"]),
            "object_yaw_range_deg": float(perturbation_cfg["yaw_deg_max"]),
            "hard success rate": (float(success_count) / float(len(all_cases))) if all_cases else None,
            "hard_success_rate": (float(success_count) / float(len(all_cases))) if all_cases else None,
            "object_xy_offset": summarize_xy_offsets(all_cases),
            "object_yaw_offset_deg": summarize_scalar([float(record["object_yaw_offset_deg"]) for record in all_cases]),
            "checkpoint": args_cli.checkpoint,
            "checkpoint_epoch": ckpt_meta["epoch"],
            "exec_horizon": int(args_cli.exec_horizon),
            "horizon": int(args_cli.horizon),
            "obs_noise_std": float(args_cli.obs_noise_std),
            "action_noise_std": float(args_cli.action_noise_std),
            "save_dir": str(save_dir),
        }

        write_json(all_cases_path, all_cases)
        write_json(failure_cases_path, failure_cases)
        write_json(summary_path, summary_payload)
        print(f"[INFO] all_cases_path={all_cases_path}", flush=True)
        print(f"[INFO] failure_cases_path={failure_cases_path}", flush=True)
        print(f"[INFO] summary_path={summary_path}", flush=True)
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
