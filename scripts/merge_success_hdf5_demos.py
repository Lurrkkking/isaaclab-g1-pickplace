#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import h5py


EXCLUDED_DATASETS = {"processed_actions"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge success-only Mimic HDF5 shards into a single demo file.")
    parser.add_argument("--input_hdf5s", type=str, nargs="+", required=True)
    parser.add_argument("--output_hdf5", type=str, required=True)
    parser.add_argument("--max_demos", type=int, default=600)
    return parser.parse_args()


def ensure_dir(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def natural_demo_keys(keys: list[str]) -> list[str]:
    def key_fn(value: str) -> tuple[int, str]:
        if value.startswith("demo_"):
            suffix = value[5:]
            if suffix.isdigit():
                return int(suffix), value
        return 10**18, value

    return sorted(keys, key=key_fn)


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def copy_group_contents(src_group, dst_group) -> None:
    copy_attrs(src_group, dst_group)
    for key, item in src_group.items():
        if key in EXCLUDED_DATASETS:
            continue
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_group_contents(item, child)
        else:
            src_group.copy(item, dst_group, name=key)


def load_summary(path: pathlib.Path) -> dict[str, Any] | None:
    summary_path = path.with_suffix(".summary.json")
    smoke_summary_path = path.with_suffix(".smoke_summary.json")
    for candidate in [summary_path, smoke_summary_path]:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    return None


def main() -> None:
    args = parse_args()
    if args.max_demos <= 0:
        raise ValueError("--max_demos must be positive")

    input_paths = [pathlib.Path(path) for path in args.input_hdf5s]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input HDF5 not found: {path}")

    output_path = pathlib.Path(args.output_hdf5)
    ensure_dir(output_path)

    merged_count = 0
    shard_records: list[dict[str, Any]] = []

    with h5py.File(output_path, "w") as out_f:
        out_data = out_f.create_group("data")

        for shard_idx, input_path in enumerate(input_paths):
            if merged_count >= args.max_demos:
                break

            summary = load_summary(input_path)
            contributed = 0
            available_success = None
            if summary is not None:
                available_success = int(summary.get("success_count", summary.get("mimic_success_count", 0)))

            with h5py.File(input_path, "r") as in_f:
                if "data" not in in_f:
                    raise KeyError(f"{input_path} is missing 'data' group")
                demo_keys = natural_demo_keys(list(in_f["data"].keys()))
                for demo_key in demo_keys:
                    if merged_count >= args.max_demos:
                        break
                    src_demo = in_f["data"][demo_key]
                    success_attr = src_demo.attrs.get("success", True)
                    try:
                        success_attr = success_attr.item()
                    except Exception:
                        pass
                    if success_attr is not True:
                        continue

                    dst_demo = out_data.create_group(f"demo_{merged_count}")
                    copy_group_contents(src_demo, dst_demo)
                    dst_demo.attrs["success"] = True
                    if "num_samples" in src_demo.attrs:
                        dst_demo.attrs["num_samples"] = src_demo.attrs["num_samples"]
                    merged_count += 1
                    contributed += 1

            shard_records.append(
                {
                    "shard_index": shard_idx,
                    "input_hdf5": str(input_path),
                    "summary_success_count": available_success,
                    "contributed_demos": contributed,
                }
            )

    summary_path = output_path.with_suffix(".merge_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_hdf5": str(output_path),
                "max_demos": int(args.max_demos),
                "merged_demo_count": merged_count,
                "input_hdf5s": [str(path) for path in input_paths],
                "shards": shard_records,
            },
            f,
            indent=2,
        )

    print(f"output_hdf5: {output_path}")
    print(f"merged_demo_count: {merged_count}")
    print(f"merge_summary: {summary_path}")
    for record in shard_records:
        print(
            f"shard={record['shard_index']} input={record['input_hdf5']} "
            f"contributed_demos={record['contributed_demos']} "
            f"summary_success_count={record['summary_success_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
