#!/bin/bash
# Setup script for environment service benchmark

# Create required directories
mkdir -p benchmark_results
mkdir -p data

echo "Generating datasets for benchmark..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Generatedataset
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/dataset_config.yaml" \
    --train_path data/sokoban-vision-benchmark/train.parquet \
    --test_path data/sokoban-vision-benchmark/test.parquet \
    --force_gen

# Run environment service benchmark

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH

# Create output directory
mkdir -p benchmark_results



# Run benchmark
echo "Running service benchmark"
python -m vagen.utils_benchmark.service_benchmark --config "$SCRIPT_DIR/benchmark_config.yaml"

echo "Benchmark complete. Results saved to benchmark_results directory."