import argparse
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=5)
parser.add_argument("--frame_dir", type=str, default="frames/g1_pickplace_viewport")
parser.add_argument("--video_path", type=str, default="videos/g1_pickplace_viewport.mp4")
parser.add_argument("--resolution_width", type=int, default=1280)
parser.add_argument("--resolution_height", type=int, default=720)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")

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

import carb
import omni.kit.app
from omni.kit.async_engine import run_coroutine
import omni.usd
from omni.kit.viewport.utility import (
    capture_viewport_to_file,
    get_active_viewport,
    next_viewport_frame_async,
)
from pxr import Usd, UsdGeom

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def iter_descendants(prim):
    for current in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        yield current


def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def get_robot_bbox(stage, robot_path="/World/envs/env_0/Robot"):
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bbox = bbox_cache.ComputeWorldBound(robot_prim)
    bbox_range = world_bbox.ComputeAlignedBox()
    mesh_count = sum(1 for prim in iter_descendants(robot_prim) if prim.GetTypeName() == "Mesh")
    return {
        "exists": True,
        "active": robot_prim.IsActive(),
        "loaded": robot_prim.IsLoaded(),
        "mesh_count": mesh_count,
        "min": tuple(bbox_range.GetMin()),
        "max": tuple(bbox_range.GetMax()),
    }


async def warmup_viewport(viewport_api, frames=4):
    app = omni.kit.app.get_app()
    for _ in range(frames):
        await app.next_update_async()
        await next_viewport_frame_async(viewport_api)


async def capture_frame(viewport_api, file_path: str):
    ensure_parent(file_path)
    await warmup_viewport(viewport_api, frames=2)
    capture = capture_viewport_to_file(viewport_api, file_path=file_path, is_hdr=False)
    await capture.wait_for_result(completion_frames=8)
    await warmup_viewport(viewport_api, frames=2)
    return os.path.isfile(file_path)


def run_coro_and_pump(coro):
    task = run_coroutine(coro)
    while not task.done():
        simulation_app.update()
    return task.result()


def capture_frame_with_retry(viewport_api, file_path: str, max_attempts: int = 4) -> bool:
    for attempt in range(1, max_attempts + 1):
        ok = run_coro_and_pump(capture_frame(viewport_api, file_path))
        if ok:
            if attempt > 1:
                print("[INFO] capture_retry_success:", file_path, "attempt", attempt)
            return True
        print("[WARN] capture_retry:", file_path, "attempt", attempt)
        simulation_app.update()
        simulation_app.update()
    return False


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
        return False
    ensure_parent(str(video_path))
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    return True


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.viewer.eye = (2.5, -2.5, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.75)
    env_cfg.viewer.resolution = (args_cli.resolution_width, args_cli.resolution_height)

    frame_dir = Path(args_cli.frame_dir)
    video_path = Path(args_cli.video_path)
    frame_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()
    print("[INFO] reset ok")

    env.unwrapped.sim.set_camera_view(
        eye=[2.5, -2.5, 1.8],
        target=[0.0, 0.0, 0.75],
    )

    viewport_api = get_active_viewport()
    if viewport_api is None:
        raise RuntimeError("Active viewport is not available.")
    viewport_api.resolution = (args_cli.resolution_width, args_cli.resolution_height)
    viewport_api.resolution_scale = 1
    viewport_api.updates_enabled = True

    run_coro_and_pump(warmup_viewport(viewport_api, frames=6))

    stage = omni.usd.get_context().get_stage()
    bbox_info = get_robot_bbox(stage)
    print("[INFO] robot_bbox:", bbox_info)

    first_frame = frame_dir / "frame_0000.png"
    ok = capture_frame_with_retry(viewport_api, str(first_frame))
    print("[INFO] captured:", str(first_frame), ok)

    action_dim = env.unwrapped.action_manager.total_action_dim
    print("[INFO] action_dim:", action_dim)

    for step_idx in range(1, max(args_cli.num_steps, 5)):
        env.unwrapped.sim.set_camera_view(
            eye=[2.5, -2.5, 1.8],
            target=[0.0, 0.0, 0.75],
        )
        actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        frame_path = frame_dir / f"frame_{step_idx:04d}.png"
        ok = capture_frame_with_retry(viewport_api, str(frame_path))
        print("[INFO] captured:", str(frame_path), ok)

    screenshot_path = frame_dir / "stage_viewport_screenshot.png"
    ok = capture_frame_with_retry(viewport_api, str(screenshot_path))
    print("[INFO] stage_screenshot:", str(screenshot_path), ok)

    env.close()

    encoded = encode_video_with_ffmpeg(frame_dir, video_path, args_cli.fps)
    if not encoded:
        encoded = encode_video_with_imageio(frame_dir, video_path, args_cli.fps)

    print("[INFO] video_encoded:", encoded)
    print("[INFO] video_path:", str(video_path))
    print("[INFO] frame_dir:", str(frame_dir))


try:
    main()
finally:
    simulation_app.close()
