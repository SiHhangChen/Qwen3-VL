#!/usr/bin/env bash

set -euo pipefail

# Queue the TS01 and WR03 MemER high-level Qwen3-VL finetunes behind the
# currently running pi0.5 job. Each task starts from the same Qwen3-VL base
# model and writes an independent checkpoint directory.

QWEN_FINETUNE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${QWEN_FINETUNE_DIR}/scripts/train_memer_qwen3vl.sh"
QWEN_REPO_DIR="$(cd -- "${QWEN_FINETUNE_DIR}/.." && pwd)"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
# PIDs of the currently running training the queue waits to free up. Override
# per-machine; leave empty to skip the PID wait.
WAIT_PIDS="${WAIT_PIDS:-508776}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${QWEN_REPO_DIR}/logs/memer_qwen3vl}"

mkdir -p "${LOG_DIR}"
printf '%s\n' "$$" > "${LOG_DIR}/queue_${RUN_TAG}.pid"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "${POLL_SECONDS}"; then
  log "POLL_SECONDS must be a positive integer, got ${POLL_SECONDS}."
  exit 2
fi
if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  log "Training launcher is missing or not executable: ${TRAIN_SCRIPT}"
  exit 1
fi

wait_for_initial_processes() {
  local pid
  local -a alive

  if [[ -z "${WAIT_PIDS}" ]]; then
    return
  fi

  log "Waiting for current training PIDs: ${WAIT_PIDS}"
  while true; do
    alive=()
    for pid in ${WAIT_PIDS//,/ }; do
      if [[ -d "/proc/${pid}" ]]; then
        alive+=("${pid}")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      log "All recorded training PIDs have exited."
      return
    fi
    log "Still running: ${alive[*]}"
    sleep "${POLL_SECONDS}"
  done
}

query_gpu_pids() {
  timeout 20s nvidia-smi -i "${GPU_IDS}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}' \
    | sort -u \
    | paste -sd, -
}

wait_for_idle_gpus() {
  local gpu_pids
  local idle_checks=0

  log "Waiting for GPUs ${GPU_IDS} to have no compute processes."
  while (( idle_checks < 2 )); do
    if gpu_pids="$(query_gpu_pids)"; then
      if [[ -z "${gpu_pids}" ]]; then
        idle_checks=$((idle_checks + 1))
        log "GPUs are idle (${idle_checks}/2 confirmations)."
      else
        idle_checks=0
        log "GPU compute processes still present: ${gpu_pids}"
      fi
    else
      idle_checks=0
      log "nvidia-smi query failed; keeping the job queued."
    fi
    if (( idle_checks < 2 )); then
      sleep "${POLL_SECONDS}"
    fi
  done
}

run_training() {
  local task_name="$1"
  local dataset="$2"
  local max_steps="$3"
  local output_dir="${QWEN_REPO_DIR}/checkpoints/${task_name}-50-stride10-v2-qwen3vl-4b"
  local training_log="${LOG_DIR}/${task_name}_50_stride10_v2_${RUN_TAG}.log"
  local status

  log "Starting ${task_name}: dataset=${dataset}, steps=${max_steps}, GPUs=${GPU_IDS}."
  log "Checkpoint directory: ${output_dir}"
  log "Training log: ${training_log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  NPROC_PER_NODE=4 \
  DATASET="${dataset}" \
  MAX_STEPS="${max_steps}" \
  OUTPUT_DIR="${output_dir}" \
  PER_DEVICE_BATCH_SIZE=4 \
  GRAD_ACCUM_STEPS=8 \
  RESUME=false \
  PYTHONUNBUFFERED=1 \
    "${TRAIN_SCRIPT}" 2>&1 | tee -a "${training_log}"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" != "0" ]]; then
    log "${task_name} training failed with status ${status}; queue stopped."
    return "${status}"
  fi
  log "${task_name} training completed successfully."
}

log "Queue tag: ${RUN_TAG}"
log "TS01 uses 762 steps and WR03 uses 2250 steps: approximately 15 epochs"
log "for each dataset at global batch size 128, matching the WA01 recipe."

wait_for_initial_processes
wait_for_idle_gpus
run_training ts01 ts01_memer_sft_50_stride10_v2 762
run_training wr03 wr03_memer_sft_50_stride10_v2 2250

log "All queued MemER Qwen3-VL training runs completed successfully."
