import argparse
import os
import subprocess
from pathlib import Path
import shutil

import gymnasium as gym
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--dataset_file", type=str, default="./datasets/generated_dataset_g1_locomanip_20.hdf5")
parser.add_argument("--demo_key", type=str, default="demo_0")
parser.add_argument("--output_dir", type=str, default="videos/g1_locomanip_demo_replay")
parser.add_argument("--resolution_width", type=int, default=1280)
parser.add_argument("--resolution_height", type=int, default=720)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--keep_frames", default=False, action="store_true")
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

import h5py
import omni.kit.app
import omni.usd
from omni.kit.async_engine import run_coroutine
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport, next_viewport_frame_async
from pxr import Usd, UsdGeom

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def ensure_parent(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def iter_descendants(prim):
    for current in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        yield current


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


def run_coro_and_pump(coro, max_updates: int = 1200):
    task = run_coroutine(coro)
    update_count = 0
    while not task.done():
        simulation_app.update()
        update_count += 1
        if update_count > max_updates:
            raise TimeoutError(f"Coroutine did not complete within {max_updates} app updates.")
    return task.result()


def capture_frame_with_retry(viewport_api, file_path: str, max_attempts: int = 4) -> bool:
    for attempt in range(1, max_attempts + 1):
        ok = run_coro_and_pump(capture_frame(viewport_api, file_path), max_updates=1800)
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


def load_demo(path: str, demo_key: str):
    with h5py.File(path, "r") as f:
        demo = f["data"][demo_key]
        actions = torch.tensor(demo["actions"][:], dtype=torch.float32)
        processed_actions = None
        if "processed_actions" in demo:
            processed_actions = torch.tensor(demo["processed_actions"][:], dtype=torch.float32)
    return actions, processed_actions


def load_initial_state(path: str, demo_key: str, device: str):
    handler = HDF5DatasetFileHandler()
    handler.open(path)
    episode = handler.load_episode(demo_key, device=device)
    handler.close()
    if episode is None:
        raise RuntimeError(f"Unable to load episode '{demo_key}' from {path}")
    initial_state = episode.get_initial_state()
    if initial_state is None:
        raise RuntimeError(f"Episode '{demo_key}' does not contain initial_state")
    return initial_state


def select_action_stream(env_action_dim: int, actions: torch.Tensor, processed_actions: torch.Tensor | None):
    print("[INFO] env action dim:", env_action_dim)
    print("[INFO] dataset actions shape:", tuple(actions.shape))
    if env_action_dim == actions.shape[1]:
        print("[INFO] replay action source: actions")
        print("[INFO] official replay behavior: raw dataset actions via EpisodeData.get_next_action()")
        return actions
    if processed_actions is not None:
        print("[INFO] dataset processed_actions shape:", tuple(processed_actions.shape))
    raise RuntimeError(
        f"Replay requires raw dataset actions whose last dimension matches env action dim. "
        f"Received env action dim {env_action_dim}, actions {tuple(actions.shape)}, "
        f"processed_actions {None if processed_actions is None else tuple(processed_actions.shape)}. "
        "Do not feed processed_actions into env.step(); they are post-processed joint targets recorded for debugging."
    )


def set_camera(env):
    env.unwrapped.sim.set_camera_view(
        eye=[3.0, -4.0, 2.2],
        target=[0.0, 0.2, 0.9],
    )


def set_camera_dynamic(env):
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    robot_root = robot.data.root_pos_w[0]
    object_root = obj.data.root_pos_w[0]
    target_x = float((robot_root[0] + object_root[0]) * 0.5)
    target_y = float((robot_root[1] + object_root[1]) * 0.5)
    target_z = max(0.9, float(object_root[2]) + 0.15)
    env.unwrapped.sim.set_camera_view(
        eye=[3.0 + target_x, -4.0 + target_y, 2.2],
        target=[target_x, target_y, target_z],
    )


def main():
    output_dir = Path(args_cli.output_dir)
    frame_dir = output_dir / ".frames_tmp"
    video_path = output_dir / "g1_locomanip_demo_replay.mp4"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.viewer.eye = (3.0, -4.0, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.2, 0.9)
    env_cfg.viewer.resolution = (args_cli.resolution_width, args_cli.resolution_height)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print("[INFO] reset ok")

    print("[INFO] loading initial_state:", args_cli.demo_key)
    initial_state = load_initial_state(args_cli.dataset_file, args_cli.demo_key, env.unwrapped.device)
    env.unwrapped.reset_to(initial_state, torch.tensor([0], device=env.unwrapped.device), is_relative=True)
    print("[INFO] reset_to initial_state ok")

    set_camera(env)

    print("[INFO] acquiring active viewport")
    viewport_api = get_active_viewport()
    if viewport_api is None:
        raise RuntimeError("Active viewport is not available.")
    viewport_api.resolution = (args_cli.resolution_width, args_cli.resolution_height)
    viewport_api.resolution_scale = 1
    viewport_api.updates_enabled = True

    print("[INFO] warming up viewport")
    run_coro_and_pump(warmup_viewport(viewport_api, frames=6), max_updates=1800)
    print("[INFO] viewport warmup ok")

    stage = omni.usd.get_context().get_stage()
    print("[INFO] robot_bbox:", get_robot_bbox(stage))

    env_action_dim = env.unwrapped.action_manager.total_action_dim
    actions, processed_actions = load_demo(args_cli.dataset_file, args_cli.demo_key)
    action_stream = select_action_stream(env_action_dim, actions, processed_actions)
    action_stream = action_stream.to(device=env.unwrapped.device)

    first_frame = frame_dir / "frame_0000.png"
    print("[INFO] capturing first frame:", str(first_frame))
    ok = capture_frame_with_retry(viewport_api, str(first_frame))
    print("[INFO] captured:", str(first_frame), ok)

    for step_idx in range(action_stream.shape[0]):
        set_camera_dynamic(env)
        action = action_stream[step_idx].view(1, -1)
        env.step(action)
        frame_path = frame_dir / f"frame_{step_idx + 1:04d}.png"
        ok = capture_frame_with_retry(viewport_api, str(frame_path))
        if step_idx == 0 or step_idx == action_stream.shape[0] - 1 or step_idx % 25 == 0:
            print("[INFO] captured:", str(frame_path), ok)

    env.close()

    encoded = encode_video_with_ffmpeg(frame_dir, video_path, args_cli.fps)
    if not encoded:
        encoded = encode_video_with_imageio(frame_dir, video_path, args_cli.fps)

    if encoded and not args_cli.keep_frames:
        for frame_path in frame_dir.glob("frame_*.png"):
            frame_path.unlink()
        frame_dir.rmdir()

    print("[INFO] video_encoded:", encoded)
    print("[INFO] video_path:", str(video_path))
    if args_cli.keep_frames:
        print("[INFO] frame_dir:", str(frame_dir))


try:
    main()
finally:
    simulation_app.close()
