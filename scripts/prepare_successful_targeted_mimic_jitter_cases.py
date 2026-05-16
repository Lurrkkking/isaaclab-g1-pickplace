#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small-jitter targeted Mimic cases around previously successful targeted cases."
    )
    parser.add_argument("--targeted_cases", type=str, required=True)
    parser.add_argument("--smoke_summary", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--jitter_per_case", type=int, default=4)
    parser.add_argument("--xy_jitter", type=float, default=0.005)
    parser.add_argument("--yaw_jitter_deg", type=float, default=2.0)
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


def main() -> None:
    args = parse_args()
    targeted_cases_payload = load_json(pathlib.Path(args.targeted_cases))
    smoke_summary = load_json(pathlib.Path(args.smoke_summary))

    source_cases = targeted_cases_payload["cases"] if isinstance(targeted_cases_payload, dict) else targeted_cases_payload
    if not isinstance(source_cases, list):
        raise TypeError(f"Expected list of targeted cases, got {type(source_cases)}")

    successful_case_indices = smoke_summary.get("successful_case_indices")
    if not isinstance(successful_case_indices, list):
        raise ValueError("smoke_summary is missing successful_case_indices")

    rng = np.random.default_rng(args.seed)
    selected_cases: list[dict[str, Any]] = []
    for case_idx in successful_case_indices:
        idx = int(case_idx)
        if idx < 0 or idx >= len(source_cases):
            raise IndexError(f"successful_case_index {idx} is out of range for {len(source_cases)} source cases")
        selected_cases.append(dict(source_cases[idx]))

    output_cases: list[dict[str, Any]] = []
    for success_src_idx, case in enumerate(selected_cases):
        base_pos = np.asarray(case["object_initial_pos"], dtype=np.float64)
        base_quat = normalize_quat(np.asarray(case["object_initial_quat"], dtype=np.float64))

        base_case = dict(case)
        base_case["source_success_case_index"] = int(successful_case_indices[success_src_idx])
        base_case["success_jitter_id"] = 0
        base_case["source_case_origin"] = "successful_targeted_case"
        output_cases.append(base_case)

        for jitter_idx in range(args.jitter_per_case):
            pos = base_pos.copy()
            pos[0] += float(rng.uniform(-args.xy_jitter, args.xy_jitter))
            pos[1] += float(rng.uniform(-args.xy_jitter, args.xy_jitter))
            yaw_delta_deg = float(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg))
            quat = normalize_quat(quat_mul(base_quat, yaw_quat_from_deg(yaw_delta_deg)))

            jitter_case = dict(case)
            jitter_case["object_initial_pos"] = [float(x) for x in pos.tolist()]
            jitter_case["object_initial_quat"] = [float(x) for x in quat.tolist()]
            jitter_case["source_success_case_index"] = int(successful_case_indices[success_src_idx])
            jitter_case["success_jitter_id"] = int(jitter_idx + 1)
            jitter_case["source_case_origin"] = "successful_targeted_case_jitter"
            output_cases.append(jitter_case)

    payload = {
        "source_targeted_cases": args.targeted_cases,
        "source_smoke_summary": args.smoke_summary,
        "successful_case_count": len(selected_cases),
        "output_targeted_case_count": len(output_cases),
        "jitter_per_case": int(args.jitter_per_case),
        "xy_jitter": float(args.xy_jitter),
        "yaw_jitter_deg": float(args.yaw_jitter_deg),
        "cases": output_cases,
    }
    write_json(pathlib.Path(args.output_json), payload)
    print(f"successful case count: {len(selected_cases)}")
    print(f"output targeted case count: {len(output_cases)}")
    print(f"output json: {args.output_json}")


if __name__ == "__main__":
    main()
