import argparse


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

import gymnasium as gym
import omni.usd
from pxr import Usd, UsdGeom

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.assets import check_file_path


def iter_descendants(prim):
    for current in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        yield current


def token_to_str(value):
    return "None" if value is None else str(value)


env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

robot_cfg = env_cfg.scene.robot
print("[INSPECT] robot usd_path:", robot_cfg.spawn.usd_path)
print("[INSPECT] robot usd_path status:", check_file_path(robot_cfg.spawn.usd_path))

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
obs, info = env.reset()

stage = omni.usd.get_context().get_stage()
robot_path = "/World/envs/env_0/Robot"
robot_prim = stage.GetPrimAtPath(robot_path)

print("[INSPECT] robot prim exists:", robot_prim.IsValid())
if robot_prim.IsValid():
    print("[INSPECT] robot prim active:", robot_prim.IsActive())
    print("[INSPECT] robot prim loaded:", robot_prim.IsLoaded())
    print("[INSPECT] robot prim type:", robot_prim.GetTypeName())
    print("[INSPECT] robot prim references:", robot_prim.GetMetadata("references"))

    mesh_prims = []
    visibility_counts = {}
    purpose_counts = {}
    type_counts = {}
    sample_prims = []
    for prim in iter_descendants(robot_prim):
        prim_type = prim.GetTypeName() or "None"
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        if len(sample_prims) < 40:
            sample_prims.append((str(prim.GetPath()), prim_type, prim.IsInstance(), prim.IsInstanceProxy()))
        if prim.GetTypeName() == "Mesh":
            mesh_prims.append(prim)
            imageable = UsdGeom.Imageable(prim)
            visibility = token_to_str(imageable.GetVisibilityAttr().Get())
            purpose = token_to_str(imageable.GetPurposeAttr().Get())
            visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1

    print("[INSPECT] mesh_count:", len(mesh_prims))
    print("[INSPECT] mesh_visibility_counts:", visibility_counts)
    print("[INSPECT] mesh_purpose_counts:", purpose_counts)
    print("[INSPECT] first_meshes:", [str(p.GetPath()) for p in mesh_prims[:10]])
    print("[INSPECT] prim_type_counts:", type_counts)
    print("[INSPECT] sample_prims:", sample_prims)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bbox = bbox_cache.ComputeWorldBound(robot_prim)
    bbox_range = world_bbox.ComputeAlignedBox()
    print("[INSPECT] world_bbox_min:", tuple(bbox_range.GetMin()))
    print("[INSPECT] world_bbox_max:", tuple(bbox_range.GetMax()))

env.close()
simulation_app.close()
