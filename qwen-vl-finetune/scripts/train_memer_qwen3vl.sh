#!/usr/bin/env bash
set -euo pipefail

QWEN_FINETUNE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_REPO_DIR="$(cd -- "${QWEN_FINETUNE_DIR}/.." && pwd)"

QWEN_ENV_DIR="${QWEN_ENV_DIR:-${HOME}/.conda/envs/memer-qwen3vl}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${QWEN_ENV_DIR}/bin/torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_STEPS="${MAX_STEPS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${QWEN_REPO_DIR}/checkpoints/wa01-memer-qwen3vl-4b}"
# The base Qwen3-VL-4B-Instruct weights path is machine-specific: set this
# explicitly (e.g. export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct) before run.
MODEL_PATH="${MODEL_PATH:-}"
DATASET="${DATASET:-wa01_memer_sft}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-64}"
MAX_PIXELS="${MAX_PIXELS:-115200}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
RESUME="${RESUME:-false}"
DRY_RUN="${DRY_RUN:-false}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
}

require_boolean() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${name} must be true or false, got: ${value}" >&2
    exit 2
  fi
}

for setting in \
  NPROC_PER_NODE MAX_STEPS PER_DEVICE_BATCH_SIZE GRAD_ACCUM_STEPS \
  MAX_PIXELS MIN_PIXELS SAVE_STEPS SAVE_TOTAL_LIMIT DATALOADER_NUM_WORKERS; do
  require_positive_integer "${setting}" "${!setting}"
done
require_boolean RESUME "${RESUME}"
require_boolean DRY_RUN "${DRY_RUN}"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun is missing or not executable: ${TORCHRUN_BIN}" >&2
  echo "Set QWEN_ENV_DIR to the conda env that provides run/torchrun." >&2
  exit 1
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is not set; export it to the Qwen3-VL-4B-Instruct weights dir." >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Qwen model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpus[@]} != NPROC_PER_NODE )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  shopt -s nullglob dotglob
  output_entries=("${OUTPUT_DIR}"/*)
  shopt -u nullglob dotglob
  if (( ${#output_entries[@]} > 0 )); then
    if [[ "${RESUME}" != "true" ]]; then
      echo "Output directory is not empty: ${OUTPUT_DIR}" >&2
      echo "Use a new OUTPUT_DIR, or set RESUME=true to resume an existing checkpoint." >&2
      exit 1
    fi
    shopt -s nullglob
    checkpoint_dirs=("${OUTPUT_DIR}"/checkpoint-*)
    shopt -u nullglob
    if (( ${#checkpoint_dirs[@]} == 0 )); then
      echo "RESUME=true, but no checkpoint-* directory exists in ${OUTPUT_DIR}." >&2
      exit 1
    fi
  elif [[ "${RESUME}" == "true" ]]; then
    echo "RESUME=true, but output directory is empty: ${OUTPUT_DIR}" >&2
    exit 1
  fi
elif [[ "${RESUME}" == "true" ]]; then
  echo "RESUME=true, but output directory does not exist: ${OUTPUT_DIR}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES

export QWEN_VL_ATTN_IMPL="flash_attention_2"

train_command=(
  "${TORCHRUN_BIN}"
  "--nproc_per_node=${NPROC_PER_NODE}"
  qwenvl/train/train_qwen.py
  --deepspeed scripts/zero3.json \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use "${DATASET}" \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm True \
  --bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 15 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --optim adamw_torch \
  --max_pixels "${MAX_PIXELS}" \
  --min_pixels "${MIN_PIXELS}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --learning_rate 6e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --ddp_find_unused_parameters False \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --report_to none
)

global_batch_size=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
echo "Resolved MemER Qwen3-VL training settings:"
echo "  dataset              : ${DATASET}"
echo "  model                : ${MODEL_PATH}"
echo "  output               : ${OUTPUT_DIR}"
echo "  GPUs                 : ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE})"
echo "  per-device / accum   : ${PER_DEVICE_BATCH_SIZE} / ${GRAD_ACCUM_STEPS}"
echo "  global batch size    : ${global_batch_size}"
echo "  optimization steps   : ${MAX_STEPS}"
echo "  resume               : ${RESUME}"
echo -n "Training command:"
printf ' %q' "${train_command[@]}"
echo

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

cd "${QWEN_FINETUNE_DIR}"
exec "${train_command[@]}"
