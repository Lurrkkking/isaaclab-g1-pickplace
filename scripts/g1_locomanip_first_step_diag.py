import argparse

import gymnasium as gym
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/root/autodl-tmp/IsaacLab/datasets/generated_dataset_g1_locomanip_20.hdf5",
)
parser.add_argument("--demo_key", type=str, default="demo_0")
parser.add_argument("--step_index", type=int, default=0)
parser.add_argument("--joint_print_count", type=int, default=10)
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

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().flatten().tolist()


def print_state(prefix: str, state: dict, joint_print_count: int):
    robot = state["articulation"]["robot"]
    object_state = state["rigid_object"]["object"]
    root_pose = robot["root_pose"][0]
    joint_position = robot["joint_position"][0]
    object_root_pose = object_state["root_pose"][0]
    print(f"[INFO] {prefix} robot_root_pose:", to_list(root_pose))
    print(f"[INFO] {prefix} base_height:", float(root_pose[2].detach().cpu()))
    print(f"[INFO] {prefix} joint_pos_first{joint_print_count}:", to_list(joint_position[:joint_print_count]))
    print(f"[INFO] {prefix} object_root_pose:", to_list(object_root_pose))


def print_state_diff(prefix: str, runtime_state: dict, dataset_state: dict):
    for asset_type, asset_name, state_name in [
        ("articulation", "robot", "root_pose"),
        ("articulation", "robot", "joint_position"),
        ("articulation", "robot", "joint_velocity"),
        ("rigid_object", "object", "root_pose"),
        ("rigid_object", "object", "root_velocity"),
    ]:
        runtime_tensor = runtime_state[asset_type][asset_name][state_name][0]
        dataset_tensor = dataset_state[asset_type][asset_name][state_name][0]
        abs_diff = (runtime_tensor - dataset_tensor).abs()
        print(
            f"[INFO] {prefix} diff {asset_type}/{asset_name}/{state_name}:"
            f" max={float(abs_diff.max().detach().cpu()):.6f}"
            f" mean={float(abs_diff.mean().detach().cpu()):.6f}"
        )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    episode = handler.load_episode(args_cli.demo_key, device=env.device)
    env_name = handler.get_env_name()
    handler.close()

    if episode is None:
        raise RuntimeError(f"Unable to load episode '{args_cli.demo_key}' from {args_cli.dataset_file}")
    if episode.get_initial_state() is None:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain initial_state")
    if "actions" not in episode.data:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain actions")
    if "states" not in episode.data:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain states")

    actions = episode.data["actions"]
    processed_actions = episode.data.get("processed_actions")
    action_dim = env.action_manager.total_action_dim

    print("[INFO] dataset env_name:", env_name)
    print("[INFO] replay task:", args_cli.task)
    print("[INFO] env action dim:", action_dim)
    print("[INFO] dataset actions shape:", tuple(actions.shape))
    if processed_actions is not None:
        print("[INFO] dataset processed_actions shape:", tuple(processed_actions.shape))
    if actions.shape[1] != action_dim:
        raise RuntimeError(
            f"Dataset actions dim {actions.shape[1]} does not match env action dim {action_dim}. "
            "This diagnostic only accepts raw actions."
        )

    env.reset()
    initial_state = episode.get_initial_state()
    env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

    restored_state = env.scene.get_state(is_relative=True)
    print_state("restored", restored_state, args_cli.joint_print_count)
    print_state_diff("reset_to_vs_initial_state", restored_state, initial_state)

    action = actions[args_cli.step_index].unsqueeze(0)
    print(f"[INFO] action[{args_cli.step_index}] shape:", tuple(action.shape))
    print(f"[INFO] action[{args_cli.step_index}] first10:", to_list(action[0, :10]))
    env.step(action)

    runtime_state = env.scene.get_state(is_relative=True)
    dataset_state = episode.get_state(args_cli.step_index)
    if dataset_state is None:
        raise RuntimeError(f"Episode '{args_cli.demo_key}' does not contain states[{args_cli.step_index}]")

    print_state("runtime_after_step", runtime_state, args_cli.joint_print_count)
    print_state("dataset_state_after_step", dataset_state, args_cli.joint_print_count)
    print_state_diff("runtime_vs_dataset_step_state", runtime_state, dataset_state)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
