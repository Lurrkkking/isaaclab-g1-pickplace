#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare random object-pose Mimic cases for G1 locomanipulation.")
    parser.add_argument("--num_cases", type=int, required=True)
    parser.add_argument("--xy_range", type=float, required=True)
    parser.add_argument("--yaw_range_deg", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--nominal_pose_reference",
        type=str,
        default="/root/autodl-tmp/IsaacLab/logs/act_failure_reset_cases_xy006_yaw20/targeted_mimic_cases.json",
        help="JSON file containing object_initial_pos/object_initial_quat plus object offsets to recover nominal pose.",
    )
    return parser.parse_args()


def ensure_dir(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return quat / norm


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


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def yaw_quat_from_deg(yaw_deg: float) -> np.ndarray:
    half = math.radians(yaw_deg) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_cases(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("cases", payload)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list-like payload, got {type(payload)}")
    return [dict(case) for case in payload]


def recover_nominal_pose(reference_case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(reference_case["object_initial_pos"], dtype=np.float64)
    quat = normalize_quat(np.asarray(reference_case["object_initial_quat"], dtype=np.float64))
    xy_offset = np.asarray(reference_case.get("object_xy_offset", [0.0, 0.0]), dtype=np.float64)
    yaw_offset_deg = float(reference_case.get("object_yaw_offset_deg", 0.0))

    nominal_pos = pos.copy()
    nominal_pos[0] -= float(xy_offset[0])
    nominal_pos[1] -= float(xy_offset[1])

    nominal_quat = normalize_quat(quat_mul(quat, quat_conjugate(yaw_quat_from_deg(yaw_offset_deg))))
    return nominal_pos, nominal_quat


def make_case(
    *,
    case_idx: int,
    seed: int,
    nominal_pos: np.ndarray,
    nominal_quat: np.ndarray,
    xy_offset: np.ndarray,
    yaw_offset_deg: float,
) -> dict[str, Any]:
    pos = nominal_pos.copy()
    pos[0] += float(xy_offset[0])
    pos[1] += float(xy_offset[1])

    quat = normalize_quat(quat_mul(nominal_quat, yaw_quat_from_deg(yaw_offset_deg)))

    return {
        "case_id": f"random_case_{case_idx:06d}",
        "base_failure_id": f"random_case_{case_idx:06d}",
        "rollout_idx": int(case_idx),
        "seed": int(seed),
        "jitter_id": 0,
        "object_xy_offset": [float(x) for x in xy_offset.tolist()],
        "object_yaw_offset_deg": float(yaw_offset_deg),
        "object_initial_pos": [float(x) for x in pos.tolist()],
        "object_initial_quat": [float(x) for x in quat.tolist()],
        "source": "random",
        "source_case_origin": "random_object_pose_sampling",
    }


def main() -> None:
    args = parse_args()

    if args.num_cases <= 0:
        raise ValueError("--num_cases must be positive")
    if args.xy_range < 0.0:
        raise ValueError("--xy_range must be >= 0")
    if args.yaw_range_deg < 0.0:
        raise ValueError("--yaw_range_deg must be >= 0")

    reference_path = pathlib.Path(args.nominal_pose_reference)
    reference_payload = load_json(reference_path)
    reference_cases = extract_cases(reference_payload)
    if not reference_cases:
        raise ValueError(f"No cases found in nominal pose reference: {reference_path}")
    nominal_pos, nominal_quat = recover_nominal_pose(reference_cases[0])

    rng = np.random.default_rng(args.seed)
    cases: list[dict[str, Any]] = []
    for case_idx in range(args.num_cases):
        xy_offset = rng.uniform(low=-args.xy_range, high=args.xy_range, size=2).astype(np.float64)
        yaw_offset_deg = float(rng.uniform(low=-args.yaw_range_deg, high=args.yaw_range_deg))
        cases.append(
            make_case(
                case_idx=case_idx,
                seed=args.seed,
                nominal_pos=nominal_pos,
                nominal_quat=nominal_quat,
                xy_offset=xy_offset,
                yaw_offset_deg=yaw_offset_deg,
            )
        )

    output_json = pathlib.Path(args.output_json)
    payload = {
        "source_reference": str(reference_path),
        "source": "random",
        "sampling_space": "object_pose_perturbation",
        "num_cases": int(args.num_cases),
        "xy_range": float(args.xy_range),
        "yaw_range_deg": float(args.yaw_range_deg),
        "seed": int(args.seed),
        "nominal_object_initial_pos": [float(x) for x in nominal_pos.tolist()],
        "nominal_object_initial_quat": [float(x) for x in nominal_quat.tolist()],
        "cases": cases,
    }
    write_json(output_json, payload)

    offsets = [case["object_xy_offset"] for case in cases]
    yaw_offsets = [float(case["object_yaw_offset_deg"]) for case in cases]
    print(f"num cases: {len(cases)}")
    print(f"xy range: {args.xy_range}")
    print(f"yaw range deg: {args.yaw_range_deg}")
    print(f"seed: {args.seed}")
    print(f"source reference: {reference_path}")
    print(f"nominal object initial pos: {payload['nominal_object_initial_pos']}")
    print(f"nominal object initial quat: {payload['nominal_object_initial_quat']}")
    print(f"object xy offset stats: {json.dumps(summarize_xy_offsets(offsets), indent=2)}")
    print(f"yaw offset deg stats: {json.dumps(summarize_scalar(yaw_offsets), indent=2)}")
    print(f"output json: {output_json}")


if __name__ == "__main__":
    main()
