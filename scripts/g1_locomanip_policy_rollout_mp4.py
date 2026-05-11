import argparse
import copy
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


parser = argparse.ArgumentParser(description="Run robomimic policy rollouts and export MP4 videos.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--norm_factor_min", type=float, default=None)
parser.add_argument("--norm_factor_max", type=float, default=None)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")

args_pre, _ = parser.parse_known_args()
if args_pre.enable_pinocchio:
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def ensure_dir(path_str: str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def set_camera(env):
    env.unwrapped.sim.set_camera_view(
        eye=[2.6, 3.2, 1.8],
        target=[0.0, 0.2, 0.95],
    )


def prepare_policy_obs(obs_dict, env):
    # This logic is copied from scripts/imitation_learning/robomimic/play.py
    obs = copy.deepcopy(obs_dict["policy"])
    for ob in obs:
        obs[ob] = torch.squeeze(obs[ob])

    if hasattr(env.cfg, "image_obs_list"):
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"].keys():
                image = torch.squeeze(obs_dict["policy"][image_name])
                image = image.permute(2, 0, 1).clone().float()
                image = image / 255.0
                image = image.clip(0.0, 1.0)
                obs[image_name] = image
    return obs


def maybe_unnormalize_actions(actions: np.ndarray) -> np.ndarray:
    # This logic is copied from scripts/imitation_learning/robomimic/play.py
    if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
        actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min
    return actions


def rollout_once(policy, env, success_term, horizon, env_device, video_path: Path):
    policy.start_episode()
    obs_dict, _ = env.reset()
    set_camera(env)

    first_frame = env.render()
    if first_frame is None:
        raise RuntimeError("env.render() returned None. This rollout video path requires render_mode='rgb_array'.")

    rollout_log = {
        "video_path": str(video_path),
        "success": False,
        "terminated": False,
        "truncated": False,
        "final_step": 0,
        "final_info": None,
        "action_shape": None,
        "env_action_dim": int(env.action_space.shape[-1]),
        "obs_keys": sorted(list(obs_dict["policy"].keys())),
    }

    with imageio.get_writer(str(video_path), fps=args_cli.fps) as writer:
        writer.append_data(first_frame)

        for step_idx in range(horizon):
            obs = prepare_policy_obs(obs_dict, env)
            actions = policy(obs)
            actions = maybe_unnormalize_actions(actions)
            actions = torch.from_numpy(actions).to(device=env_device).view(1, env.action_space.shape[1])

            rollout_log["action_shape"] = list(actions.shape)

            obs_dict, _, terminated, truncated, info = env.step(actions)
            frame = env.render()
            if frame is None:
                raise RuntimeError(f"env.render() returned None at step {step_idx}")
            writer.append_data(frame)

            success = bool(success_term.func(env, **success_term.params)[0])
            done = bool(terminated[0]) if isinstance(terminated, torch.Tensor) else bool(terminated)
            timeout = bool(truncated[0]) if isinstance(truncated, torch.Tensor) else bool(truncated)

            if success or done or timeout:
                rollout_log["success"] = success
                rollout_log["terminated"] = done
                rollout_log["truncated"] = timeout
                rollout_log["final_step"] = step_idx + 1
                rollout_log["final_info"] = str(info)
                return rollout_log

    rollout_log["final_step"] = horizon
    rollout_log["final_info"] = "reached_horizon"
    return rollout_log


def main():
    output_dir = Path(args_cli.output_dir)
    ensure_dir(str(output_dir))

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    env_cfg.viewer.eye = (2.6, 3.2, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.95)

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Keep the policy device selection aligned with the official play.py behavior.
    policy_device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=policy_device)

    logs = []
    for rollout_idx in range(args_cli.num_rollouts):
        print(f"[INFO] Starting rollout {rollout_idx}")
        video_path = output_dir / f"rollout_{rollout_idx}.mp4"
        rollout_log = rollout_once(policy, env, success_term, args_cli.horizon, env.device, video_path)
        rollout_log["rollout_index"] = rollout_idx
        logs.append(rollout_log)
        print(
            f"[INFO] Rollout {rollout_idx}: success={rollout_log['success']} "
            f"terminated={rollout_log['terminated']} truncated={rollout_log['truncated']} "
            f"final_step={rollout_log['final_step']}"
        )

    summary_path = output_dir / "rollout_summary.json"
    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_rollouts": args_cli.num_rollouts,
        "horizon": args_cli.horizon,
        "fps": args_cli.fps,
        "norm_factor_min": args_cli.norm_factor_min,
        "norm_factor_max": args_cli.norm_factor_max,
        "results": logs,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    success_count = sum(1 for item in logs if item["success"])
    print(f"[INFO] Successful rollouts: {success_count}/{len(logs)}")
    print(f"[INFO] Summary saved to: {summary_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
