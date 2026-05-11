import argparse
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=None)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/root/autodl-tmp/IsaacLab/datasets/generated_dataset_g1_locomanip_20.hdf5",
)
parser.add_argument("--demo_key", type=str, default="demo_0")
parser.add_argument("--output", type=str, default="videos/g1_locomanip_demo_0.mp4")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--resolution_width", type=int, default=1280)
parser.add_argument("--resolution_height", type=int, default=720)
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

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def encode_video_with_ffmpeg(frame_dir: Path, video_path: Path, fps: int) -> bool:
    ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    ffmpeg_path = ffmpeg.stdout.strip()
    if ffmpeg.returncode != 0 or not ffmpeg_path:
        return False
    ensure_parent(str(video_path))
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARN] ffmpeg failed:")
        print(result.stderr)
        return False
    return True


def encode_video_with_imageio(frame_dir: Path, video_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio unavailable: {exc}")
        return False

    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        print("[WARN] no frames found for imageio encoding")
        return False

    ensure_parent(str(video_path))
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    return True


def save_frame(path: Path, frame: np.ndarray):
    import imageio.v2 as imageio

    ensure_parent(str(path))
    imageio.imwrite(path, frame)


def load_episode(dataset_file: str, demo_key: str, device: str):
    handler = HDF5DatasetFileHandler()
    handler.open(dataset_file)
    env_name = handler.get_env_name()
    episode = handler.load_episode(demo_key, device=device)
    handler.close()
    return env_name, episode


def set_camera(env):
    env.unwrapped.sim.set_camera_view(
        eye=[2.6, 3.2, 1.8],
        target=[0.0, 0.2, 0.95],
    )


def main():
    env_name_from_dataset, episode = load_episode(args_cli.dataset_file, args_cli.demo_key, args_cli.device)
    if episode is None:
        raise RuntimeError(f"Unable to load episode '{args_cli.demo_key}' from {args_cli.dataset_file}")
    if episode.get_initial_state() is None:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain initial_state")
    if "actions" not in episode.data:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain actions")

    replay_task = args_cli.task or env_name_from_dataset
    if replay_task != env_name_from_dataset:
        raise RuntimeError(
            f"Dataset was recorded with env '{env_name_from_dataset}', but replay task was forced to '{replay_task}'. "
            "For generated G1 locomanip demos, replay in the dataset's Mimic env to avoid divergence."
        )

    env_cfg = parse_env_cfg(replay_task, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.viewer.eye = (2.6, 3.2, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.95)
    env_cfg.viewer.resolution = (args_cli.resolution_width, args_cli.resolution_height)

    env = gym.make(replay_task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    env.unwrapped.reset_to(
        episode.get_initial_state(),
        torch.tensor([0], device=env.unwrapped.device),
        is_relative=True,
    )
    set_camera(env)

    actions = episode.data["actions"]
    if actions.shape[1] != env.unwrapped.action_manager.total_action_dim:
        raise RuntimeError(
            f"Dataset actions dim {actions.shape[1]} does not match env action dim "
            f"{env.unwrapped.action_manager.total_action_dim}."
        )

    total_steps = actions.shape[0] if args_cli.max_steps < 0 else min(args_cli.max_steps, actions.shape[0])
    output_path = Path(args_cli.output)
    frame_dir = output_path.parent / f".{output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame0 = env.render()
    if frame0 is None:
        raise RuntimeError("env.render() returned None. This replay path requires render_mode='rgb_array'.")
    save_frame(frame_dir / "frame_0000.png", frame0)
    print("[INFO] captured frame 0")

    for step_idx in range(total_steps):
        action = actions[step_idx].view(1, -1)
        env.step(action)
        frame = env.render()
        if frame is None:
            raise RuntimeError(f"env.render() returned None at step {step_idx}")
        save_frame(frame_dir / f"frame_{step_idx + 1:04d}.png", frame)
        if step_idx == 0 or step_idx == total_steps - 1 or step_idx % 25 == 0:
            print(f"[INFO] captured frame {step_idx + 1}")

    env.close()

    encoded = encode_video_with_ffmpeg(frame_dir, output_path, args_cli.fps)
    if not encoded:
        encoded = encode_video_with_imageio(frame_dir, output_path, args_cli.fps)
    if not encoded:
        raise RuntimeError("Unable to encode MP4 with ffmpeg or imageio.")

    print("[INFO] video_path:", str(output_path))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
