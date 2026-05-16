#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal targeted Mimic smoke test for G1 locomanipulation.")
parser.add_argument("--task", type=str, default="Isaac-Locomanipulation-G1-Abs-Mimic-v0")
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--targeted_cases", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--num_cases", type=int, default=1)
parser.add_argument("--start_case", type=int, default=0)
parser.add_argument("--max_success_demos", type=int, default=None)
parser.add_argument("--success_only", action="store_true", default=False)
parser.add_argument("--max_abs_xy_offset", type=float, default=None)
parser.add_argument("--max_abs_yaw_offset_deg", type=float, default=None)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
parser.add_argument("--keep_failed", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_mimic.envs  # noqa: F401,E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401,E402

from isaaclab_mimic.datagen.data_generator import DataGenerator  # noqa: E402
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool  # noqa: E402
from isaaclab_mimic.datagen.generation import setup_env_config  # noqa: E402


def tensor_to_list(value: torch.Tensor) -> list[float]:
    return [float(x) for x in value.detach().float().cpu().tolist()]


def load_targeted_cases(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise TypeError(f"Expected list of targeted cases, got {type(cases)}")
    indexed_cases = []
    for case_idx, case in enumerate(cases):
        case_copy = dict(case)
        case_copy["targeted_case_index"] = case_idx
        indexed_cases.append(case_copy)
    return indexed_cases


def case_passes_filters(case: dict[str, Any]) -> bool:
    if args_cli.max_abs_xy_offset is not None:
        xy_offset = case.get("object_xy_offset", [0.0, 0.0])
        if any(abs(float(x)) > args_cli.max_abs_xy_offset for x in xy_offset):
            return False
    if args_cli.max_abs_yaw_offset_deg is not None:
        if abs(float(case.get("object_yaw_offset_deg", 0.0))) > args_cli.max_abs_yaw_offset_deg:
            return False
    return True


def set_object_default_pose(env: ManagerBasedRLMimicEnv, case: dict[str, Any]) -> None:
    rigid_object = env.scene["object"]
    root_state = rigid_object.data.default_root_state.clone()
    root_state[0, 0:3] = torch.tensor(case["object_initial_pos"], dtype=root_state.dtype, device=root_state.device)
    root_state[0, 3:7] = torch.tensor(case["object_initial_quat"], dtype=root_state.dtype, device=root_state.device)
    root_state[0, 7:13] = 0.0
    rigid_object.data.default_root_state[:] = root_state


def get_object_pose_from_env(env: ManagerBasedRLMimicEnv) -> dict[str, list[float]]:
    rigid_object = env.scene["object"]
    root_state = rigid_object.data.root_state_w[0]
    return {
        "pos": tensor_to_list(root_state[:3]),
        "quat": tensor_to_list(root_state[3:7]),
        "vel": tensor_to_list(root_state[7:13]),
    }


def get_object_pose_from_scene_state(scene_state: dict[str, Any]) -> dict[str, list[float]]:
    obj = scene_state["rigid_object"]["object"]
    return {
        "pos": tensor_to_list(obj["root_pose"][0, :3]),
        "quat": tensor_to_list(obj["root_pose"][0, 3:7]),
        "vel": tensor_to_list(obj["root_velocity"][0]),
    }


async def run_one_case(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term,
    case: dict[str, Any],
) -> dict[str, Any]:
    set_object_default_pose(env, case)
    default_pose = {
        "pos": tensor_to_list(env.scene["object"].data.default_root_state[0, :3]),
        "quat": tensor_to_list(env.scene["object"].data.default_root_state[0, 3:7]),
    }
    print(
        f"[DEBUG] targeted_pose pos={case['object_initial_pos']} quat={case['object_initial_quat']}",
        flush=True,
    )
    print(f"[DEBUG] default_root_state_after_injection={default_pose}", flush=True)
    result = await data_generator.generate(
        env_id=0,
        success_term=success_term,
        env_reset_queue=env_reset_queue,
        env_action_queue=env_action_queue,
        pause_subtask=False,
        export_demo=True,
        motion_planner=None,
    )
    return result


def drive_env_until_done(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    event_loop: asyncio.AbstractEventLoop,
    task: asyncio.Task,
) -> Any:
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    with torch.inference_mode():
        while not task.done():
            while env_action_queue.qsize() != env.num_envs and not task.done():
                event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    reset_pose = get_object_pose_from_env(env)
                    reset_scene_state = env.scene.get_state(is_relative=True)
                    reset_state_pose = get_object_pose_from_scene_state(reset_scene_state)
                    print(f"[DEBUG] env.reset actual_object_pose={reset_pose}", flush=True)
                    print(f"[DEBUG] scene.get_state after reset object_pose={reset_state_pose}", flush=True)
                    env_reset_queue.task_done()

            if task.done():
                break

            actions = torch.zeros(env.action_space.shape, device=env.device)
            for _ in range(env.num_envs):
                env_id, action = event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action.to(env.device)
            env.step(actions)
            for _ in range(env.num_envs):
                env_action_queue.task_done()

            if env.sim.is_stopped():
                break

    return event_loop.run_until_complete(task)


def main() -> None:
    target_cases = load_targeted_cases(pathlib.Path(args_cli.targeted_cases))
    if not target_cases:
        raise ValueError("No targeted cases found.")

    filtered_cases = [case for case in target_cases if case_passes_filters(case)]
    if args_cli.start_case < 0:
        raise ValueError("--start_case must be >= 0")
    if args_cli.start_case >= len(filtered_cases):
        raise ValueError(
            f"--start_case={args_cli.start_case} is out of range for {len(filtered_cases)} filtered targeted cases."
        )

    selected_cases = filtered_cases[args_cli.start_case :]
    if args_cli.num_cases > 0:
        selected_cases = selected_cases[: args_cli.num_cases]
    if not selected_cases:
        raise ValueError("No targeted cases selected after applying filters and start_case/num_cases.")

    output_file = pathlib.Path(args_cli.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    keep_failed = bool(args_cli.keep_failed)
    if args_cli.success_only:
        keep_failed = False

    env_cfg, success_term = setup_env_config(
        env_name=args_cli.task,
        output_dir=str(output_file.parent),
        output_file_name=output_file.stem,
        num_envs=1,
        device=args_cli.device,
        generation_num_trials=len(selected_cases),
    )
    env_cfg.datagen_config.generation_keep_failed = keep_failed
    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise TypeError(f"Expected ManagerBasedRLMimicEnv, got {type(env)}")

    env.reset()

    event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    info_pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=asyncio.Lock())
    info_pool.load_from_dataset_file(args_cli.input_file)
    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=info_pool)

    smoke_records = []
    total_attempted_cases = 0
    mimic_success_count = 0
    mimic_failed_count = 0
    successful_case_indices: list[int] = []
    failed_case_indices: list[int] = []
    try:
        for case_idx, case in enumerate(selected_cases):
            if args_cli.max_success_demos is not None and mimic_success_count >= args_cli.max_success_demos:
                print(
                    f"[INFO] reached max_success_demos={args_cli.max_success_demos}; stopping early after "
                    f"{total_attempted_cases} attempts",
                    flush=True,
                )
                break

            print(
                f"[INFO] targeted_case={case_idx} global_index={case['targeted_case_index']} "
                f"base_failure_id={case['base_failure_id']} jitter_id={case['jitter_id']}",
                flush=True,
            )
            task = event_loop.create_task(run_one_case(env, env_reset_queue, env_action_queue, data_generator, success_term, case))
            result = drive_env_until_done(env, env_reset_queue, env_action_queue, event_loop, task)
            initial_state_pose = get_object_pose_from_scene_state(result["initial_state"])
            print(f"[DEBUG] recorded_initial_state object_pose={initial_state_pose}", flush=True)
            replay_reset_pose = None
            try:
                env.reset_to(result["initial_state"], torch.tensor([0], device=env.device), is_relative=True)
                replay_reset_pose = get_object_pose_from_env(env)
                print(f"[DEBUG] replay_reset_to initial_state object_pose={replay_reset_pose}", flush=True)
            except Exception as exc:
                replay_reset_pose = {"error": str(exc)}
                print(f"[WARNING] replay_reset_to initial_state failed: {exc}", flush=True)

            total_attempted_cases += 1
            case_success = bool(result["success"])
            if case_success:
                mimic_success_count += 1
                successful_case_indices.append(int(case["targeted_case_index"]))
            else:
                mimic_failed_count += 1
                failed_case_indices.append(int(case["targeted_case_index"]))

            smoke_records.append(
                {
                    "case_index": case_idx,
                    "targeted_case_index": int(case["targeted_case_index"]),
                    "base_failure_id": case["base_failure_id"],
                    "rollout_idx": int(case["rollout_idx"]),
                    "seed": int(case["seed"]),
                    "jitter_id": case["jitter_id"],
                    "success": case_success,
                    "source_failure_reason": case.get("source_failure_reason"),
                    "object_xy_offset": case.get("object_xy_offset"),
                    "object_yaw_offset_deg": case.get("object_yaw_offset_deg"),
                    "targeted_object_initial_pos": case["object_initial_pos"],
                    "targeted_object_initial_quat": case["object_initial_quat"],
                    "min_left_object_dist": case.get("min_left_object_dist"),
                    "object_motion_range": case.get("object_motion_range"),
                    "recorded_initial_state_object_pose": initial_state_pose,
                    "replay_reset_object_pose": replay_reset_pose,
                }
            )
            print(
                f"[INFO] targeted_case={case_idx} global_index={case['targeted_case_index']} success={case_success} "
                f"mimic_success_count={mimic_success_count} mimic_failed_count={mimic_failed_count}",
                flush=True,
            )
    finally:
        smoke_summary_path = output_file.with_suffix(".smoke_summary.json")
        with smoke_summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "output_file": str(output_file),
                    "targeted_cases_file": str(args_cli.targeted_cases),
                    "input_source_file": str(args_cli.input_file),
                    "success_only": bool(args_cli.success_only),
                    "keep_failed": keep_failed,
                    "start_case": int(args_cli.start_case),
                    "num_cases": int(args_cli.num_cases),
                    "max_success_demos": args_cli.max_success_demos,
                    "max_abs_xy_offset": args_cli.max_abs_xy_offset,
                    "max_abs_yaw_offset_deg": args_cli.max_abs_yaw_offset_deg,
                    "total_filtered_cases": len(filtered_cases),
                    "total_selected_cases": len(selected_cases),
                    "total_attempted_cases": total_attempted_cases,
                    "mimic_success_count": mimic_success_count,
                    "mimic_failed_count": mimic_failed_count,
                    "successful_case_indices": successful_case_indices,
                    "failed_case_indices": failed_case_indices,
                    "cases": smoke_records,
                },
                f,
                indent=2,
            )
        print(f"[INFO] smoke_summary={smoke_summary_path}", flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
