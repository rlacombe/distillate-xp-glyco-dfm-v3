# Glyco DFM v3: Publication-Ready Discrete Flow Matching

## Overview

Training a 50M non-autoregressive model for glycan structure prediction using proper Discrete Flow Matching (DFM).

**Status:** v3 corrects data leakage in v2, implements proper train/val/test split.

## Data

- **Source:** CandyCrush v2 (Zenodo)
- **Training:** X_train.pkl with 80/20 train/val split
- **Testing:** X_test.pkl (full, ~50k samples) — directly comparable to GlycoBART
- **Baseline:** GlycoBART 0.8903

## Algorithm

**Discrete Flow Matching (Gat et al., NeurIPS 2024)**
- Training: continuous t ∈ [0,1], masked corruption
- Inference: Euler ODE solver with flexible step counts
- Evaluation: Pareto frontier (accuracy vs steps)

## Model

- 50M parameters (d=512, 4L enc, 8L dec, 8 heads)
- Batch size: 128
- LR: 3e-4 → 9e-6 floor, cosine decay
- 80/20 train/val split on X_train
- TRAIN_SECONDS=7200 (estimated 50k+ steps)

## Checkpointing

- **best_model.pt:** saved on every improvement (top1_structural_accuracy)
- **latest_model.pt:** saved every 1000 steps (resume point)
- **checkpoint_step_XXXXXX.pt:** periodic snapshots every 5000 steps (keep last 3)
- Saved to persistent experiment directory: `/Users/romain/experiments/glyco-dfm-v3/checkpoints/`

## Goals

1. Beat GlycoBART baseline (0.8903)
2. Demonstrate non-autoregressive speedup (96× faster)
3. Build Pareto frontier (accuracy/speed trade-off)
4. Publication-ready results with proper evaluation
5. Robust, resumable training with checkpointing

## Success Criteria

- **N=1:** accuracy ≥ 0.8987 (one-shot validation)
- **N=8-16:** accuracy ≥ 0.92 (target multi-step)
- Full test set evaluation on X_test.pkl (no holdout)
- All checkpoints saved and resumable
