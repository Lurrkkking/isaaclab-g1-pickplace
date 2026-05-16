#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Mimic HDF5 consistency for initial_state/object pose/action semantics.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--reference_file", type=str, default=None)
    parser.add_argument("--demo_key", type=str, default="demo_0")
    return parser.parse_args()


def stat_dict(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
    }


def print_json(title: str, payload: Any) -> None:
    print(title)
    print(json.dumps(payload, indent=2))


def list_demo_structure(file_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"file": str(file_path), "demo_keys": [], "demos": {}}
    with h5py.File(file_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{file_path} does not contain 'data'")
        demo_keys = sorted(f["data"].keys())
        summary["demo_keys"] = demo_keys
        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            fields = sorted(demo.keys())
            entry: dict[str, Any] = {
                "fields": fields,
                "has_actions": "actions" in demo,
                "has_processed_actions": "processed_actions" in demo,
                "has_obs": "obs" in demo,
                "has_states": "states" in demo,
                "has_initial_state": "initial_state" in demo,
                "attrs": {k: (v.item() if hasattr(v, "item") else v) for k, v in demo.attrs.items()},
            }
            if "actions" in demo:
                actions = demo["actions"][...]
                entry["actions"] = stat_dict(actions)
                if actions.ndim == 2:
                    entry["action_dim"] = int(actions.shape[1])
                    entry["episode_length"] = int(actions.shape[0])
            if "processed_actions" in demo:
                entry["processed_actions"] = stat_dict(demo["processed_actions"][...])
            if "initial_state" in demo:
                entry["initial_state_keys"] = sorted(demo["initial_state"].keys())
                if "rigid_object" in demo["initial_state"]:
                    entry["initial_state_rigid_object_keys"] = sorted(demo["initial_state"]["rigid_object"].keys())
            summary["demos"][demo_key] = entry
    return summary


def extract_object_state(demo_group) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "initial_state" in demo_group and "rigid_object" in demo_group["initial_state"]:
        rigid_object = demo_group["initial_state"]["rigid_object"]
        if "object" in rigid_object:
            obj = rigid_object["object"]
            result["initial_state_object_root_pose"] = obj["root_pose"][...].tolist()
            result["initial_state_object_root_velocity"] = obj["root_velocity"][...].tolist()
            result["initial_state_object_fields"] = sorted(obj.keys())
    if "states" in demo_group and "rigid_object" in demo_group["states"] and "object" in demo_group["states"]["rigid_object"]:
        obj_state = demo_group["states"]["rigid_object"]["object"]
        result["states_object_fields"] = sorted(obj_state.keys())
        if "root_pose" in obj_state and len(obj_state["root_pose"]) > 0:
            result["first_state_object_root_pose"] = obj_state["root_pose"][0].tolist()
        if "root_velocity" in obj_state and len(obj_state["root_velocity"]) > 0:
            result["first_state_object_root_velocity"] = obj_state["root_velocity"][0].tolist()
    return result


def inspect_demo(file_path: Path, demo_key: str) -> dict[str, Any]:
    with h5py.File(file_path, "r") as f:
        demo = f["data"][demo_key]
        actions = demo["actions"][...]
        payload: dict[str, Any] = {
            "file": str(file_path),
            "demo_key": demo_key,
            "episode_length": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "first_action_stats": stat_dict(actions[:1]),
            "all_action_stats": stat_dict(actions),
            "first_action": actions[0].tolist(),
            "first_10_actions": actions[:10].tolist(),
            "contains_processed_actions": "processed_actions" in demo,
        }
        if "processed_actions" in demo:
            payload["processed_action_dim"] = int(demo["processed_actions"].shape[1])
            payload["processed_action_stats"] = stat_dict(demo["processed_actions"][...])
        payload.update(extract_object_state(demo))
        return payload


def compare_demo_shapes(target: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "action_dim",
        "episode_length",
        "contains_processed_actions",
        "processed_action_dim",
        "initial_state_object_fields",
        "states_object_fields",
    ]
    comparison = {}
    for key in keys:
        comparison[key] = {
            "target": target.get(key),
            "reference": reference.get(key),
            "match": target.get(key) == reference.get(key),
        }
    comparison["first_action_range"] = {
        "target": target["first_action_stats"],
        "reference": reference["first_action_stats"],
    }
    comparison["all_action_range"] = {
        "target": target["all_action_stats"],
        "reference": reference["all_action_stats"],
    }
    comparison["initial_state_object_root_pose"] = {
        "target": target.get("initial_state_object_root_pose"),
        "reference": reference.get("initial_state_object_root_pose"),
    }
    return comparison


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    reference_file = Path(args.reference_file) if args.reference_file else None

    target_structure = list_demo_structure(input_file)
    print_json("=== HDF5 Structure: Input ===", target_structure)

    target_demo = inspect_demo(input_file, args.demo_key)
    print_json("=== Target Demo Detail ===", target_demo)

    if reference_file is not None:
        reference_structure = list_demo_structure(reference_file)
        print_json("=== HDF5 Structure: Reference ===", reference_structure)
        reference_demo = inspect_demo(reference_file, args.demo_key)
        print_json("=== Reference Demo Detail ===", reference_demo)
        comparison = compare_demo_shapes(target_demo, reference_demo)
        print_json("=== Target vs Reference Comparison ===", comparison)

    if target_demo["contains_processed_actions"]:
        print("[INFO] processed_actions exists in targeted HDF5. Replay must use raw actions, not processed_actions.")


if __name__ == "__main__":
    main()
