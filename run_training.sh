#!/bin/bash
# Quick training launcher

set -e

echo "🚀 Starting CRT v2 Training..."
echo ""

# Activate environment
eval "$(conda shell.bash hook)"
conda activate crt_v2

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ GPU not detected!"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run training
python train_crt_v2.py

echo ""
echo "✅ Training finished!"
