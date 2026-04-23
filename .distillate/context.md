# Prior Run History

This experiment has **1 prior run(s)**. Below are the most recent 1.

**IMPORTANT:** Review this history before starting. Build on what worked, avoid repeating failed approaches. Reference specific run IDs when explaining your reasoning.

### xp-glyco-v3-001 [completed]
**Key metric to maximize:** `top1_structural_accuracy`. Current best: **top1_structural_accuracy=0.8602** (from xp-glyco-v3-001).
Your goal is to maximize `top1_structural_accuracy` across runs. Report this metric for every run.

**Hypothesis:** Publication-ready DFM with proper 80/20 train/val split on X_train (no data leakage) will demonstrate correct train/test separation and generate Pareto frontier comparable to GlycoBART
**Changes:** Initial demo: validate checkpointing, train/val split, Pareto frontier
**Results:** val_token_acc=0.825, best_val_token_acc=0.825, top1_structural_accuracy=0.8602, best_top1_structural_accuracy=0.8602, n_params=50123456, train_steps=5000, n_train=10000, n_val=2500
