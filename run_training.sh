#!/bin/bash
set -e

cd /Users/romain/experiments/glyco-dfm-v3

echo "=== Glyco DFM v3 Experimentalist Agent ==="
echo "Starting training at $(date)"
echo ""

# Use Distillate dev venv which has all dependencies
PYTHON="/Users/romain/Code/Distillate/distillate-dev/.venv/bin/python"

# Run training
$PYTHON train.py

echo ""
echo "Training complete at $(date)"
