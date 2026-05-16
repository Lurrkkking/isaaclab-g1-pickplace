#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ACT failure cases into targeted Mimic generation cases.")
    parser.add_argument("--failure_cases", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--jitter_per_case", type=int, default=3)
    parser.add_argument("--xy_jitter", type=float, default=0.01)
    parser.add_argument("--yaw_jitter_deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: Any) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_scalar(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarize_xy_offsets(offsets: list[list[float]]) -> dict[str, Any]:
    if not offsets:
        return {
            "x": {"min": None, "max": None, "mean": None},
            "y": {"min": None, "max": None, "mean": None},
            "norm": {"min": None, "max": None, "mean": None},
        }
    arr = np.asarray(offsets, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=1)
    return {
        "x": summarize_scalar(arr[:, 0].tolist()),
        "y": summarize_scalar(arr[:, 1].tolist()),
        "norm": summarize_scalar(norm.tolist()),
    }


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_quat_from_deg(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return quat / norm


def is_valid_failure_case(case: dict[str, Any]) -> bool:
    pos = case.get("object_initial_pos")
    quat = case.get("object_initial_quat")
    return isinstance(pos, list) and len(pos) == 3 and isinstance(quat, list) and len(quat) == 4


def make_targeted_case(
    *,
    case: dict[str, Any],
    base_failure_id: str,
    jitter_id: int,
    object_initial_pos: np.ndarray,
    object_initial_quat: np.ndarray,
) -> dict[str, Any]:
    return {
        "base_failure_id": base_failure_id,
        "rollout_idx": int(case["rollout_idx"]),
        "seed": int(case["seed"]),
        "object_initial_pos": [float(x) for x in object_initial_pos.tolist()],
        "object_initial_quat": [float(x) for x in object_initial_quat.tolist()],
        "object_xy_offset": [float(x) for x in case["object_xy_offset"]],
        "object_yaw_offset_deg": float(case["object_yaw_offset_deg"]),
        "jitter_id": int(jitter_id),
        "source_failure_reason": case.get("failure_reason"),
        "min_left_object_dist": case.get("min_left_object_dist"),
        "object_motion_range": case.get("object_motion_range"),
    }


def main() -> None:
    args = parse_args()
    failure_cases_path = pathlib.Path(args.failure_cases)
    output_json_path = pathlib.Path(args.output_json)

    failure_cases = load_json(failure_cases_path)
    if not isinstance(failure_cases, list):
        raise TypeError(f"Expected a list in {failure_cases_path}, got {type(failure_cases)}")

    input_failure_count = len(failure_cases)
    valid_failure_cases = [case for case in failure_cases if is_valid_failure_case(case)]
    if args.max_cases is not None:
        valid_failure_cases = valid_failure_cases[: args.max_cases]

    rng = np.random.default_rng(args.seed)
    targeted_cases: list[dict[str, Any]] = []
    for valid_idx, case in enumerate(valid_failure_cases):
        base_failure_id = f"failure_rollout_{int(case['rollout_idx']):04d}"
        base_pos = np.asarray(case["object_initial_pos"], dtype=np.float64)
        base_quat = normalize_quat(np.asarray(case["object_initial_quat"], dtype=np.float64))
        targeted_cases.append(
            make_targeted_case(
                case=case,
                base_failure_id=base_failure_id,
                jitter_id=0,
                object_initial_pos=base_pos,
                object_initial_quat=base_quat,
            )
        )

        for jitter_idx in range(args.jitter_per_case):
            pos = base_pos.copy()
            pos[0] += float(rng.uniform(-args.xy_jitter, args.xy_jitter))
            pos[1] += float(rng.uniform(-args.xy_jitter, args.xy_jitter))
            yaw_delta_deg = float(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg))
            quat = normalize_quat(quat_mul(base_quat, yaw_quat_from_deg(yaw_delta_deg)))
            targeted_cases.append(
                make_targeted_case(
                    case=case,
                    base_failure_id=base_failure_id,
                    jitter_id=jitter_idx + 1,
                    object_initial_pos=pos,
                    object_initial_quat=quat,
                )
            )

    payload = {
        "source_failure_cases": str(failure_cases_path),
        "input_failure_count": input_failure_count,
        "valid_failure_count": len(valid_failure_cases),
        "output_targeted_case_count": len(targeted_cases),
        "jitter_per_case": int(args.jitter_per_case),
        "xy_jitter": float(args.xy_jitter),
        "yaw_jitter_deg": float(args.yaw_jitter_deg),
        "cases": targeted_cases,
    }
    write_json(output_json_path, payload)

    valid_offsets = [[float(x) for x in case["object_xy_offset"]] for case in valid_failure_cases]
    valid_yaw_offsets = [float(case["object_yaw_offset_deg"]) for case in valid_failure_cases]
    print(f"input failure count: {input_failure_count}")
    print(f"valid failure count: {len(valid_failure_cases)}")
    print(f"output targeted case count: {len(targeted_cases)}")
    print(f"object xy offset stats: {json.dumps(summarize_xy_offsets(valid_offsets), indent=2)}")
    print(f"yaw offset deg stats: {json.dumps(summarize_scalar(valid_yaw_offsets), indent=2)}")
    print(f"output json: {output_json_path}")


if __name__ == "__main__":
    main()
