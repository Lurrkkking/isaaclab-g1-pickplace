#!/usr/bin/env python3
import argparse
import glob
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
import types

DIFFUSION_POLICY_ROOT = "/root/autodl-tmp/diffusion_policy"
if DIFFUSION_POLICY_ROOT not in sys.path:
    sys.path.insert(0, DIFFUSION_POLICY_ROOT)

ROBODIFF_SITE_PACKAGES = sorted(
    glob.glob("/root/autodl-tmp/.conda_pkgs/envs/robodiff/lib/python*/site-packages")
)
for site_packages in ROBODIFF_SITE_PACKAGES:
    if site_packages not in sys.path:
        sys.path.append(site_packages)

import dill
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

try:
    import zarr  # noqa: F401
except Exception:
    zarr_stub = types.ModuleType("zarr")

    class _ZarrArray:
        pass

    zarr_stub.Array = _ZarrArray
    sys.modules["zarr"] = zarr_stub


OBS_KEYS = [
    "left_eef_pos",
    "left_eef_quat",
    "right_eef_pos",
    "right_eef_quat",
    "hand_joint_state",
    "object",
]


parser = argparse.ArgumentParser(description="Minimal IsaacLab rollout for G1 low-dim diffusion policy.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/root/autodl-tmp/diffusion_policy/data/outputs/g1_locomanip_lowdim_10ep/checkpoints/latest.ckpt",
)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--horizon", type=int, default=50)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--record_video", action="store_true", default=False)
parser.add_argument("--video_dir", type=str, default="videos/diffusion_rollout")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--verbose_steps", action="store_true", default=False)
parser.add_argument("--exec_horizon", type=int, default=8)
parser.add_argument("--n_obs_steps", type=int, default=None)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--print_dataset_stats", action="store_true", default=False)
parser.add_argument("--print_motion_stats", action="store_true", default=False)

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

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (  # noqa: E402
    TrainDiffusionUnetLowdimWorkspace,
)


VIEWPORT_CAPTURE_READY = False


def ensure_dir(path_str):
    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)


def nested_keys(obj):
    if hasattr(obj, "keys"):
        return sorted(str(key) for key in obj.keys())
    return [str(type(obj))]


def unwrap_policy_obs(obs):
    if not hasattr(obs, "keys"):
        raise TypeError(f"Observation container has no keys: {type(obs)}")
    if "policy" in obs:
        return obs["policy"]
    return obs


def flatten_policy_obs(policy_obs):
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


def tensor_stats_str(tensor):
    tensor = tensor.detach().float().cpu()
    return (
        f"shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f}"
    )


def load_policy(checkpoint_path, device):
    ckpt_path = pathlib.Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDiffusionUnetLowdimWorkspace(payload["cfg"])
    workspace.load_payload(payload)
    cfg = workspace.cfg

    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) else workspace.model
    if policy is None:
        policy = workspace.model
    policy.to(device)
    policy.eval()
    return workspace, policy


def print_dataset_obs_stats(dataset_path, sample_frames=5):
    if dataset_path is None:
        print("[INFO] dataset_path: unavailable")
        return
    if not hasattr(zarr, "open"):
        print(f"[INFO] dataset_path: {dataset_path} (zarr unavailable)")
        return

    root = zarr.open(dataset_path, mode="r")
    obs_arr = root["data"]["obs"]
    total_frames = int(obs_arr.shape[0])
    sample_frames = min(sample_frames, total_frames)
    sample = np.asarray(obs_arr[:sample_frames], dtype=np.float32)
    print(f"[INFO] dataset_path: {dataset_path}")
    print(f"[INFO] dataset obs sample shape: {tuple(sample.shape)}")
    print(f"[INFO] dataset obs sample stats: {tensor_stats_str(torch.from_numpy(sample))}")
    for idx in range(sample_frames):
        frame = torch.from_numpy(np.asarray(obs_arr[idx], dtype=np.float32))
        print(f"[INFO] dataset obs frame[{idx}] stats: {tensor_stats_str(frame)}")


def tensor_flag_to_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def set_camera(env):
    env.unwrapped.sim.set_camera_view(
        eye=[3.0, 4.0, 2.2],
        target=[0.0, 0.2, 0.9],
    )


def ensure_viewport_capture_imports():
    global VIEWPORT_CAPTURE_READY
    if VIEWPORT_CAPTURE_READY:
        return

    import omni.kit.app  # noqa: F401
    from omni.kit.async_engine import run_coroutine  # noqa: F401
    from omni.kit.viewport.utility import (  # noqa: F401
        capture_viewport_to_file,
        get_active_viewport,
        next_viewport_frame_async,
    )

    VIEWPORT_CAPTURE_READY = True


def maybe_get_object_target(success_term, object_pos):
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
    target_z = object_pos[2].item()
    if "max_height" in params:
        target_z = min(target_z, float(params["max_height"]))
    return torch.tensor([target_x, target_y, target_z], dtype=object_pos.dtype)


def extract_motion_state(policy_obs, success_term):
    left_eef_pos = policy_obs.get("left_eef_pos")
    right_eef_pos = policy_obs.get("right_eef_pos")
    object_pos = policy_obs.get("object_pos")
    if left_eef_pos is None or right_eef_pos is None or object_pos is None:
        return None

    left_eef_pos = left_eef_pos[0].detach().cpu()
    right_eef_pos = right_eef_pos[0].detach().cpu()
    object_pos = object_pos[0].detach().cpu()
    target_pos = maybe_get_object_target(success_term, object_pos)
    if target_pos is not None:
        target_pos = target_pos.detach().cpu()

    return {
        "left_eef_pos": left_eef_pos,
        "right_eef_pos": right_eef_pos,
        "object_pos": object_pos,
        "target_pos": target_pos,
    }


def maybe_print_motion_stats(env_step_idx, env_action, prev_env_action, policy_obs, success_term):
    if not args_cli.print_motion_stats:
        return
    if env_step_idx % 50 != 0:
        return

    state = extract_motion_state(policy_obs, success_term)
    action_stats = (
        f"min={env_action.min().item():.6f} "
        f"max={env_action.max().item():.6f} "
        f"mean={env_action.mean().item():.6f}"
    )
    if prev_env_action is None:
        action_delta_mean_abs = 0.0
    else:
        action_delta_mean_abs = (env_action - prev_env_action).abs().mean().item()

    print(f"motion stats step {env_step_idx}: action {action_stats}")
    print(f"motion stats step {env_step_idx}: action_delta_mean_abs={action_delta_mean_abs:.6f}")
    if state is None:
        print(f"motion stats step {env_step_idx}: left_eef_pos=unavailable right_eef_pos=unavailable object_pos=unavailable")
        print(f"motion stats step {env_step_idx}: object_target_distance=unavailable")
        return

    print(
        f"motion stats step {env_step_idx}: "
        f"left_eef_pos={state['left_eef_pos'].tolist()} "
        f"right_eef_pos={state['right_eef_pos'].tolist()} "
        f"object_pos={state['object_pos'].tolist()}"
    )
    if state["target_pos"] is None:
        print(f"motion stats step {env_step_idx}: object_target_distance=unavailable")
    else:
        object_target_distance = torch.norm(state["object_pos"] - state["target_pos"]).item()
        print(
            f"motion stats step {env_step_idx}: "
            f"object_target_distance={object_target_distance:.6f} "
            f"target_pos={state['target_pos'].tolist()}"
        )


class RolloutVideoRecorder:
    def __init__(self, env, video_dir, rollout_idx, fps, enabled):
        self.env = env
        self.enabled = enabled
        self.fps = fps
        self.video_dir = pathlib.Path(video_dir)
        self.rollout_idx = rollout_idx
        self.video_path = self.video_dir / f"rollout_{rollout_idx}.mp4"
        self.mode = None
        self.writer = None
        self.viewport_api = None
        self.frame_dir = self.video_dir / f".rollout_{rollout_idx}_frames"
        self.frame_idx = 0

    def start(self):
        if not self.enabled:
            return
        ensure_dir(str(self.video_dir))

        frame = self.env.render()
        if frame is not None:
            self.mode = "rgb_array"
            self.writer = imageio.get_writer(str(self.video_path), fps=self.fps)
            self.writer.append_data(frame)
            print(f"[INFO] video recorder mode=rgb_array path={self.video_path}")
            return

        ensure_viewport_capture_imports()
        from omni.kit.viewport.utility import get_active_viewport

        self.mode = "viewport"
        ensure_dir(str(self.frame_dir))
        self.viewport_api = get_active_viewport()
        if self.viewport_api is None:
            raise RuntimeError("Active viewport is not available for viewport capture fallback.")
        self.viewport_api.updates_enabled = True
        self._capture_viewport_frame()
        print(f"[INFO] video recorder mode=viewport path={self.video_path}")

    def capture_step(self):
        if not self.enabled:
            return
        if self.mode == "rgb_array":
            frame = self.env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None while rgb_array recording was active.")
            self.writer.append_data(frame)
        elif self.mode == "viewport":
            self._capture_viewport_frame()

    def finish(self):
        if not self.enabled:
            return
        if self.mode == "rgb_array" and self.writer is not None:
            self.writer.close()
            print(f"[INFO] saved video: {self.video_path}")
            return
        if self.mode == "viewport":
            self._encode_viewport_frames()
            print(f"[INFO] saved video: {self.video_path}")

    def _capture_viewport_frame(self):
        frame_path = self.frame_dir / f"frame_{self.frame_idx:04d}.png"
        ok = self._capture_frame_with_retry(str(frame_path))
        if not ok:
            raise RuntimeError(f"Viewport capture failed for frame: {frame_path}")
        self.frame_idx += 1

    def _encode_viewport_frames(self):
        frame_paths = sorted(self.frame_dir.glob("frame_*.png"))
        if not frame_paths:
            raise RuntimeError(f"No viewport frames found in {self.frame_dir}")
        with imageio.get_writer(str(self.video_path), fps=self.fps) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        shutil.rmtree(self.frame_dir, ignore_errors=True)

    def _capture_frame_with_retry(self, file_path, max_attempts=4):
        for attempt in range(1, max_attempts + 1):
            ok = self._run_coro_and_pump(self._capture_frame(file_path))
            if ok:
                return True
            print(f"[WARN] capture_retry path={file_path} attempt={attempt}")
            simulation_app.update()
            simulation_app.update()
        return False

    async def _warmup_viewport(self, frames=2):
        import omni.kit.app
        from omni.kit.viewport.utility import next_viewport_frame_async

        app = omni.kit.app.get_app()
        for _ in range(frames):
            await app.next_update_async()
            await next_viewport_frame_async(self.viewport_api)

    async def _capture_frame(self, file_path):
        from omni.kit.viewport.utility import capture_viewport_to_file

        await self._warmup_viewport(frames=2)
        capture = capture_viewport_to_file(self.viewport_api, file_path=file_path, is_hdr=False)
        await capture.wait_for_result(completion_frames=8)
        await self._warmup_viewport(frames=2)
        return os.path.isfile(file_path)

    def _run_coro_and_pump(self, coro):
        from omni.kit.async_engine import run_coroutine

        task = run_coroutine(coro)
        while not task.done():
            simulation_app.update()
        return task.result()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device(args_cli.device)
    workspace, policy = load_policy(args_cli.checkpoint, device=device)
    cfg = workspace.cfg
    checkpoint_horizon = int(getattr(cfg.policy, "horizon", getattr(cfg, "horizon", 16)))
    checkpoint_n_obs_steps = int(getattr(cfg.policy, "n_obs_steps", getattr(cfg, "n_obs_steps", 1)))
    checkpoint_n_action_steps = int(getattr(cfg.policy, "n_action_steps", getattr(cfg, "n_action_steps", 1)))
    checkpoint_num_inference_steps = int(
        getattr(
            cfg.policy,
            "num_inference_steps",
            getattr(cfg, "num_inference_steps", getattr(policy, "num_inference_steps", 1)),
        )
    )
    obs_history_len = int(args_cli.n_obs_steps if args_cli.n_obs_steps is not None else checkpoint_n_obs_steps)
    action_dim = int(getattr(cfg.policy, "action_dim", getattr(cfg, "action_dim", 32)))

    if args_cli.num_inference_steps is not None:
        policy.num_inference_steps = int(args_cli.num_inference_steps)

    dataset_cfg = getattr(getattr(cfg, "task", None), "dataset", None)
    dataset_path = getattr(dataset_cfg, "dataset_path", None)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = None
    env_cfg.viewer.eye = (3.0, 4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.9)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None

    render_mode = "rgb_array" if args_cli.record_video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode).unwrapped
    env.seed(0)

    print(f"checkpoint path: {args_cli.checkpoint}")
    print(f"env action_space: {env.action_space}")
    print(
        "[INFO] checkpoint policy params: "
        f"horizon={checkpoint_horizon} "
        f"n_obs_steps={checkpoint_n_obs_steps} "
        f"n_action_steps={checkpoint_n_action_steps} "
        f"num_inference_steps={checkpoint_num_inference_steps}"
    )
    print(f"[INFO] rollout obs_history_len={obs_history_len}")
    print(f"[INFO] rollout final num_inference_steps={policy.num_inference_steps}")
    if args_cli.print_dataset_stats:
        t0 = time.time()
        print_dataset_obs_stats(dataset_path)
        print(f"[INFO] dataset stats elapsed_sec={time.time() - t0:.3f}")

    trial_results = []
    try:
        for rollout_idx in range(args_cli.num_rollouts):
            t_reset = time.time()
            obs, info = env.reset()
            print(f"[INFO] env.reset elapsed_sec={time.time() - t_reset:.3f}")
            set_camera(env)
            print(f"rollout {rollout_idx} reset obs keys: {nested_keys(obs)}")

            video_recorder = RolloutVideoRecorder(
                env=env,
                video_dir=args_cli.video_dir,
                rollout_idx=rollout_idx,
                fps=args_cli.fps,
                enabled=args_cli.record_video,
            )
            video_recorder.start()

            policy_obs = unwrap_policy_obs(obs)
            print(f"rollout {rollout_idx} policy obs keys: {nested_keys(policy_obs)}")

            flat_obs = flatten_policy_obs(policy_obs)
            print(f"rollout {rollout_idx} flat obs shape: {tuple(flat_obs.shape)}")
            print(f"rollout {rollout_idx} flat obs stats: {tensor_stats_str(flat_obs)}")
            normalized_obs = policy.normalizer["obs"].normalize(flat_obs.to(device))
            print(f"rollout {rollout_idx} normalized obs stats: {tensor_stats_str(normalized_obs)}")

            obs_history = flat_obs.unsqueeze(1).repeat(1, obs_history_len, 1)
            print(f"rollout {rollout_idx} obs history shape: {tuple(obs_history.shape)}")
            success_step = None
            end_step = None
            rollout_success = False
            rollout_terminated = False
            rollout_truncated = False
            env_step_idx = 0
            prev_env_action = None

            for step_idx in range(args_cli.horizon):
                if env_step_idx >= args_cli.horizon:
                    break
                with torch.no_grad():
                    result = policy.predict_action({"obs": obs_history.to(device)})

                if not isinstance(result, dict):
                    raise TypeError(f"predict_action should return dict, got {type(result)}")

                if "action" not in result:
                    raise KeyError(f'predict_action result missing "action": {result.keys()}')

                action_chunk = result["action"]
                if args_cli.exec_horizon < 1:
                    raise ValueError(f"exec_horizon must be >= 1, got {args_cli.exec_horizon}")
                exec_horizon = min(args_cli.exec_horizon, action_chunk.shape[1])
                if step_idx == 0:
                    print(f"rollout {rollout_idx} action_chunk shape: {tuple(action_chunk.shape)}")
                if args_cli.verbose_steps:
                    print(f"rollout {rollout_idx} predict_action keys: {sorted(result.keys())}")
                    print(f"rollout {rollout_idx} predicted action shape: {tuple(action_chunk.shape)}")

                for exec_idx in range(exec_horizon):
                    if env_step_idx >= args_cli.horizon:
                        break
                    action = action_chunk[:, exec_idx, :]
                    if action.shape[-1] != action_dim:
                        raise ValueError(
                            f"Action last dim should be {action_dim}, got {tuple(action.shape)}"
                        )
                    if args_cli.verbose_steps:
                        print(
                            f"rollout {rollout_idx} policy_step {step_idx} exec_step {exec_idx} action stats: "
                            f"min={action.min().item():.6f} max={action.max().item():.6f} mean={action.mean().item():.6f}"
                        )

                    env_action = action.to(env.device)
                    if env_action.ndim != 2 or env_action.shape[1] != 32:
                        raise ValueError(f"env.step action must be (num_envs, 32), got {tuple(env_action.shape)}")

                    current_env_step = env_step_idx
                    obs, reward, terminated, truncated, info = env.step(env_action)
                    video_recorder.capture_step()

                    policy_obs = unwrap_policy_obs(obs)
                    flat_obs = flatten_policy_obs(policy_obs)
                    obs_history = torch.cat([obs_history[:, 1:, :], flat_obs.unsqueeze(1)], dim=1)

                    success = False
                    if success_term is not None:
                        success = tensor_flag_to_bool(success_term.func(env, **success_term.params))
                    terminated_flag = tensor_flag_to_bool(terminated)
                    truncated_flag = tensor_flag_to_bool(truncated)
                    done = terminated_flag or truncated_flag
                    reward_value = float(reward.reshape(-1)[0]) if isinstance(reward, torch.Tensor) else reward
                    if args_cli.verbose_steps:
                        print(
                            f"rollout {rollout_idx} env_step {current_env_step}: "
                            f"reward={reward_value} terminated={terminated_flag} "
                            f"truncated={truncated_flag} success={success} done={done}"
                        )

                    maybe_print_motion_stats(
                        env_step_idx=current_env_step,
                        env_action=env_action.detach().cpu(),
                        prev_env_action=None if prev_env_action is None else prev_env_action.detach().cpu(),
                        policy_obs=policy_obs,
                        success_term=success_term,
                    )
                    prev_env_action = env_action.clone()

                    if success:
                        success_step = current_env_step
                        end_step = current_env_step
                        rollout_success = True
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        print(f"rollout {rollout_idx} success at step {current_env_step}")
                        break

                    if done:
                        end_step = current_env_step
                        rollout_terminated = terminated_flag
                        rollout_truncated = truncated_flag
                        break

                    env_step_idx = current_env_step + 1

                if rollout_success or rollout_terminated or rollout_truncated:
                    break
            if end_step is None:
                end_step = env_step_idx - 1 if env_step_idx > 0 else None
            video_recorder.finish()
            trial_results.append(rollout_success)
            print(
                f"rollout {rollout_idx} summary: success={rollout_success}, "
                f"success_step={success_step}, end_step={end_step}, "
                f"terminated={rollout_terminated}, truncated={rollout_truncated}"
            )
    finally:
        env.close()
    success_count = sum(trial_results)
    print(f"Successful trials: {success_count}, out of {len(trial_results)} trials")
    print(f"Success rate: {success_count}/{len(trial_results)}")
    print(f"Trial Results: {trial_results}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
