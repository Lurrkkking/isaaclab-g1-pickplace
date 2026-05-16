#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/IsaacLab"
CHECKPOINT="/root/autodl-tmp/act_g1/outputs/g1_act_100ep/checkpoints/epoch_0500.pt"
ANNOTATED_DATASET="/root/autodl-tmp/IsaacLab/datasets/dataset_annotated_g1_locomanip.hdf5"
BASE_CASES_DIR="/root/autodl-tmp/IsaacLab/logs/act_failure_reset_cases"
BASE_TARGETED_CASES="${BASE_CASES_DIR}/targeted_mimic_cases.json"
BASE_SUCCESS_SUMMARY="/root/autodl-tmp/IsaacLab/datasets/generated_targeted_mimic_success_filtered.smoke_summary.json"
SUMMARY_TSV="/root/autodl-tmp/IsaacLab/logs/targeted_mimic_sweep_summary.tsv"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/.conda_pkgs/envs/isaacsim-5.1.0
export TERM=xterm

cd "${ROOT}"

run_collect() {
    local save_dir="$1"
    local xy="$2"
    local yaw="$3"
    ./isaaclab.sh -p scripts/collect_act_failure_reset_cases.py \
        --headless \
        --enable_pinocchio \
        --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
        --checkpoint "${CHECKPOINT}" \
        --num_rollouts 100 \
        --horizon 400 \
        --exec_horizon 4 \
        --perturb_level medium \
        --object_xy_range "${xy}" \
        --object_yaw_range_deg "${yaw}" \
        --seed 42 \
        --save_dir "${save_dir}"
}

run_prepare() {
    local failure_json="$1"
    local output_json="$2"
    python scripts/prepare_targeted_mimic_cases.py \
        --failure_cases "${failure_json}" \
        --output_json "${output_json}" \
        --jitter_per_case 3 \
        --xy_jitter 0.01 \
        --yaw_jitter_deg 5.0 \
        --seed 42
}

run_mimic() {
    local targeted_cases="$1"
    local output_hdf5="$2"
    ./isaaclab.sh -p scripts/run_targeted_mimic_smoke.py \
        --headless \
        --enable_pinocchio \
        --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
        --input_file "${ANNOTATED_DATASET}" \
        --targeted_cases "${targeted_cases}" \
        --output_file "${output_hdf5}" \
        --num_cases 1000000 \
        --max_success_demos 100 \
        --success_only
}

append_summary_row() {
    local setting="$1"
    local summary_json="$2"
    local targeted_json="$3"
    local mimic_summary_json="$4"
    local output_hdf5="$5"
    python - <<PY
import json
from pathlib import Path

setting = ${setting@Q}
summary_json = Path(${summary_json@Q})
targeted_json = Path(${targeted_json@Q})
mimic_summary_json = Path(${mimic_summary_json@Q})
output_hdf5 = ${output_hdf5@Q}
summary_tsv = Path(${SUMMARY_TSV@Q})

collect_summary = json.loads(summary_json.read_text())
targeted_payload = json.loads(targeted_json.read_text())
mimic_summary = json.loads(mimic_summary_json.read_text())

failures = int(collect_summary["failure_count"])
targeted_cases = int(targeted_payload["output_targeted_case_count"])
success = int(mimic_summary["mimic_success_count"])
attempted = int(mimic_summary["total_attempted_cases"])
rate = (success / attempted) if attempted else 0.0

if not summary_tsv.exists():
    summary_tsv.write_text("setting\tACT rollout failures\ttargeted cases\tMimic success demos\tMimic success rate\toutput_hdf5\n")
with summary_tsv.open("a", encoding="utf-8") as f:
    f.write(f"{setting}\t{failures}\t{targeted_cases}\t{success}\t{rate:.4f}\t{output_hdf5}\n")
PY
}

run_group() {
    local setting="$1"
    local xy="$2"
    local yaw="$3"
    local dir_suffix="$4"
    local save_dir="/root/autodl-tmp/IsaacLab/logs/${dir_suffix}"
    local targeted_json="${save_dir}/targeted_mimic_cases.json"
    local output_hdf5="/root/autodl-tmp/IsaacLab/datasets/generated_targeted_mimic_success_${dir_suffix#act_failure_reset_cases_}.hdf5"
    local mimic_summary_json="${output_hdf5%.hdf5}.smoke_summary.json"

    run_collect "${save_dir}" "${xy}" "${yaw}"
    run_prepare "${save_dir}/failure_cases.json" "${targeted_json}"
    run_mimic "${targeted_json}" "${output_hdf5}"
    append_summary_row "${setting}" "${save_dir}/summary.json" "${targeted_json}" "${mimic_summary_json}" "${output_hdf5}"
}

python scripts/prepare_successful_targeted_mimic_jitter_cases.py \
    --targeted_cases "${BASE_TARGETED_CASES}" \
    --smoke_summary "${BASE_SUCCESS_SUMMARY}" \
    --output_json "/root/autodl-tmp/IsaacLab/logs/act_failure_reset_cases/targeted_mimic_cases_from_success_jitter.json" \
    --jitter_per_case 4 \
    --xy_jitter 0.005 \
    --yaw_jitter_deg 2.0 \
    --seed 42

./isaaclab.sh -p scripts/run_targeted_mimic_smoke.py \
    --headless \
    --enable_pinocchio \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file "${ANNOTATED_DATASET}" \
    --targeted_cases "/root/autodl-tmp/IsaacLab/logs/act_failure_reset_cases/targeted_mimic_cases_from_success_jitter.json" \
    --output_file "/root/autodl-tmp/IsaacLab/datasets/generated_targeted_mimic_success_from_success_jitter.hdf5" \
    --num_cases 1000000 \
    --max_success_demos 100 \
    --success_only

run_group "medium_plus_1" "0.05" "15" "act_failure_reset_cases_xy005_yaw15"
run_group "medium_plus_2" "0.06" "20" "act_failure_reset_cases_xy006_yaw20"
run_group "medium_plus_3" "0.07" "25" "act_failure_reset_cases_xy007_yaw25"

python - <<PY
import json
from pathlib import Path

summary_tsv = Path(${SUMMARY_TSV@Q})
jitter_summary = Path("/root/autodl-tmp/IsaacLab/datasets/generated_targeted_mimic_success_from_success_jitter.smoke_summary.json")
if jitter_summary.exists():
    payload = json.loads(jitter_summary.read_text())
    attempted = int(payload["total_attempted_cases"])
    success = int(payload["mimic_success_count"])
    rate = (success / attempted) if attempted else 0.0
    with summary_tsv.open("a", encoding="utf-8") as f:
        f.write(
            "success_jitter\t-\t{targeted}\t{success}\t{rate:.4f}\t{output}\n".format(
                targeted=payload["total_selected_cases"],
                success=success,
                rate=rate,
                output=payload["output_file"],
            )
        )
PY

echo "[INFO] summary_tsv=${SUMMARY_TSV}"
