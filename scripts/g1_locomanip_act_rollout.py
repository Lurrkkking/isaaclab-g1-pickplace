#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import random
import sys
import traceback
import types
from typing import Dict, Optional

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
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if args_cli.exec_horizon < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {args_cli.exec_horizon}")

    device = torch.device(args_cli.device)
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
    env.seed(0)

    trial_results = []
    try:
        for rollout_idx in range(args_cli.num_rollouts):
            print("[INFO] Before env.reset", flush=True)
            obs, info = env.reset()
            print("[INFO] reset ok", flush=True)
            set_camera(env)

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

            policy_obs = unwrap_policy_obs(obs)
            flat_obs = flatten_policy_obs(policy_obs).to(device)
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
                    flat_obs = flatten_policy_obs(policy_obs).to(device)
                    obs_history = torch.cat([obs_history[:, 1:, :], flat_obs.unsqueeze(1)], dim=1)

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
                end_step = env_step_idx - 1 if env_step_idx > 0 else None
            video_recorder.finish()
            action_trace_recorder.finish()
            trial_results.append(rollout_success)
            print(
                f"rollout {rollout_idx} summary: success={rollout_success}, "
                f"success_step={success_step}, end_step={end_step}, "
                f"terminated={rollout_terminated}, truncated={rollout_truncated}",
                flush=True,
            )
        success_count = sum(trial_results)
        print(f"Successful trials: {success_count}, out of {len(trial_results)} trials", flush=True)
        print(f"Success rate: {success_count}/{len(trial_results)}", flush=True)
        print(f"Trial Results: {trial_results}", flush=True)
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
