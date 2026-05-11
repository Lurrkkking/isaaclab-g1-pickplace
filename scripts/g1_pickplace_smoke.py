import argparse
import torch
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
)
parser.add_argument("--num_envs", type=int, default=1)

# 关键：官方部分脚本也是手动加这个参数，不是 AppLauncher 默认参数
parser.add_argument(
    "--enable_pinocchio",
    default=False,
    action="store_true",
    help="Enable Pinocchio before launching Isaac Sim.",
)

# 如果启用 pinocchio，必须在 AppLauncher 启动前先 import
# 这是为了避免 Isaac Sim runtime 加载后污染 pinocchio / hppfcl / eigenpy 这类 C++ binding
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

# 必须在 AppLauncher 启动 Isaac Sim runtime 之后显式 import pick_place 包
# 因为 pick_place 被 isaaclab_tasks 的自动导入黑名单排除了
import isaaclab_tasks.manager_based.manipulation.pick_place
import isaaclab_tasks.manager_based.locomanipulation.pick_place
from isaaclab_tasks.utils import parse_env_cfg

print("[INFO] Registered PickPlace/G1 tasks:")
for k in sorted(gym.registry.keys()):
    if "PickPlace" in k or "G1" in k:
        print("  ", k)

env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

env = gym.make(args_cli.task, cfg=env_cfg)

obs, info = env.reset()
print("[INFO] reset ok")
print("[INFO] action_space:", env.action_space)

action_dim = env.unwrapped.action_manager.total_action_dim
print("[INFO] action_dim:", action_dim)

for i in range(20):
    actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
    if i % 5 == 0:
        print(f"[INFO] step {i} ok, actions.shape={tuple(actions.shape)}")

env.close()
simulation_app.close()
print("[INFO] smoke test finished")
