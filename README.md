# glyco-dfm-v3

Publication-ready Discrete Flow Matching for glycan structure prediction.

An autonomous ML experiment powered by [Distillate](https://github.com/rlacombe/distillate).

## What is Distillate?

Distillate is an open-source tool that helps scientists design, launch, and track autonomous ML experiments — with a paper library built in. Nicolas, the research alchemist, orchestrates Claude Code agents that iteratively improve your models.

## What is glyco-dfm-v3?

**glyco-dfm-v3** corrects data leakage in v2 by implementing a proper train/val/test split:
- **Training:** X_train.pkl with 80/20 train/val split
- **Testing:** X_test.pkl (full, ~50k samples, no holdout)
- **Baseline:** GlycoBART 0.8903 on the same test set

Uses proper Discrete Flow Matching (Gat et al., NeurIPS 2024) with ODE solver inference and Pareto frontier evaluation.

## Reproducing this experiment

```bash
# Install Distillate
pip install distillate

# Clone and run
git clone https://github.com/$(gh api user -q .login)/distillate-xp-glyco-dfm-v3.git
cd distillate-xp-glyco-dfm-v3
distillate launch  # Resume the experiment
```

## Results

See `.distillate/runs.jsonl` for the full experiment history.

Output structure:
- `checkpoints/` — training checkpoints (best, latest, periodic)
- `models/` — exported inference models and results
