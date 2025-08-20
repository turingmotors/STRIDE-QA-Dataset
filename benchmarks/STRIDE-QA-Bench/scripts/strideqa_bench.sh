#!/bin/bash
#SBATCH --job-name=strideqa_bench
#SBATCH --time=2:00:00
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=sbatch_logs/strideqa_bench/%x-%j.out
#SBATCH --error=sbatch_logs/strideqa_bench/%x-%j.out

set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
STRIDEQA_BENCH_DIR=$ROOT_DIR/benchmarks/STRIDE-QA-Bench
STRIDEQA_BENCH_DATA_DIR="${STRIDEQA_BENCH_DIR}/data/STRIDE-QA-Bench"

export PYTHONPATH=$PYTHONPATH:$STRIDEQA_BENCH_DIR


# Define input directories as an array
INPUT_DIRS=(
    # "/path/to/your/results"
)

# Process each input directory
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    echo "=========================================="
    echo "Processing: $INPUT_DIR"
    echo "=========================================="

    # Check if directory exists
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "Warning: Directory $INPUT_DIR does not exist. Skipping..."
        continue
    fi

    # Check if directory contains required JSON files
    if ! ls "$INPUT_DIR"/*_t*.json 1> /dev/null 2>&1; then
        echo "Warning: No *_t*.json files found in $INPUT_DIR. Skipping..."
        continue
    fi

    OUTPUT_DIR=$INPUT_DIR

    echo "Running benchmark evaluation..."
    uv run $STRIDEQA_BENCH_DIR/src/strideqa_bench/benchmark.py \
        --input-dir $INPUT_DIR \
        --output-dir $OUTPUT_DIR \
        --annotation-dir "${STRIDEQA_BENCH_DATA_DIR}/annotation_files" \
        --config-path "$STRIDEQA_BENCH_DIR/config/tolerance.yaml"

    echo "Completed: $INPUT_DIR"
    echo ""
done

echo "=========================================="
echo "All benchmark evaluations completed!"
echo "=========================================="
