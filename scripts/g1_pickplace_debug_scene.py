import argparse
import torch
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
parser.add_argument("--num_envs", type=int, default=1)
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

import isaaclab_tasks.manager_based.manipulation.pick_place
import isaaclab_tasks.manager_based.locomanipulation.pick_place
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

env = gym.make(args_cli.task, cfg=env_cfg)
obs, info = env.reset()

print("\n[DEBUG] scene keys:")
try:
    print(env.unwrapped.scene.keys())
except Exception as e:
    print("scene.keys failed:", e)

print("\n[DEBUG] available scene entities:")
try:
    for name, obj in env.unwrapped.scene._entities.items():
        print("entity:", name, "type:", type(obj))
except Exception as e:
    print("entities failed:", e)

print("\n[DEBUG] robot info:")
try:
    robot = env.unwrapped.scene["robot"]
    print("robot:", robot)
    print("robot prim_path:", robot.cfg.prim_path)
    print("robot root_pos_w:", robot.data.root_pos_w[0].detach().cpu().numpy())
    print("robot root_quat_w:", robot.data.root_quat_w[0].detach().cpu().numpy())
    print("num joints:", len(robot.joint_names))
    print("first joints:", robot.joint_names[:10])
    print("num bodies:", len(robot.body_names))
    print("first bodies:", robot.body_names[:20])
except Exception as e:
    print("robot debug failed:", repr(e))

print("\n[DEBUG] object info:")
for key in ["object", "rigid_object", "table", "robot"]:
    try:
        ent = env.unwrapped.scene[key]
        print(key, ent)
        if hasattr(ent, "data") and hasattr(ent.data, "root_pos_w"):
            print(key, "root_pos_w:", ent.data.root_pos_w[0].detach().cpu().numpy())
    except Exception:
        pass

env.close()
simulation_app.close()
