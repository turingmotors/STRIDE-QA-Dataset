#!/bin/bash
#SBATCH --job-name=inference_internvl
#SBATCH --time=2:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.out

set -euo pipefail

TIME_START=$(date +%s)

PROJECT_DIR="$(git rev-parse --show-toplevel)"
STRIDEQA_BENCH_DIR="${PROJECT_DIR}/benchmarks/STRIDE-QA-Bench"
STRIDEQA_BENCH_DATA_DIR="${STRIDEQA_BENCH_DIR}/data/STRIDE-QA-Bench"

# Common runtime setup
source "${STRIDEQA_BENCH_DIR}/scripts/common/runtime_env.sh"

# Override defaults if needed
# export CUDA_MODULE_VERSION=12.4
# export USE_MODULES_DEFAULT=auto    # auto|true|false
export ENABLE_DETERMINISM=1
setup_runtime_env

# Model & IO setup
MODEL_PATH="${MODEL_PATH:-/data/models/InternVL/InternVL2_5-8B}"  # https://huggingface.co/OpenGVLab/InternVL2_5-8B
STAMP="$(date "+%Y%m%d_%H%M%S")"
OUTPUT_DIR="${STRIDEQA_BENCH_DIR}/results/InternVL2_5-8B-${STAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "${MODEL_PATH:-}" > "${OUTPUT_DIR}/checkpoint.txt" || true

# Inference
uv run torchrun --nproc-per-node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${STRIDEQA_BENCH_DIR}/src/strideqa_bench/inference/models/inference_internvl2_5.py" \
  --annotation-dir "${STRIDEQA_BENCH_DATA_DIR}/annotation_files" \
  --image-folder   "${STRIDEQA_BENCH_DATA_DIR}" \
  --output-dir     "${OUTPUT_DIR}" \
  --model-path     "${MODEL_PATH}" \
  --use-flash-attn True \
  --save-video     True \
  --input-size     448 \
  --max-new-tokens 200 \
  --seed 42 \
  --limit -1

# Evaluation
uv run "${STRIDEQA_BENCH_DIR}/src/strideqa_bench/benchmark.py" \
  --input-dir  "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --annotation-dir "${STRIDEQA_BENCH_DATA_DIR}/annotation_files" \
  --config-path "${STRIDEQA_BENCH_DIR}/config/tolerance.yaml"

echo "[INFO] Done evaluation. Check the results in ${OUTPUT_DIR}"

TIME_END=$(date +%s)
ELAPSED=$(( TIME_END - TIME_START ))
printf "Time elapsed: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
