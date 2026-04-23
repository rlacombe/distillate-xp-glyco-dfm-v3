# Steering: Glyco DFM v3 with Proper DFM and Checkpointing

## Objective
Train a 50M Discrete Flow Matching model for glycan structure prediction with:
- Proper train/val/test split (X_train for training, full X_test for evaluation)
- Robust checkpointing to persistent storage
- DFM ODE solver with Pareto frontier evaluation

## Dataset
- **Train:** X_train.pkl with 80/20 train/val split (NOT v2's 90/10)
- **Test:** X_test.pkl (full, ~50k samples)
- **Baseline:** GlycoBART 0.8903 on same test set

## Architecture
- 50M params (d=512, 4L enc, 8L dec, 8 heads)
- Batch size: 128
- LR: 3e-4 → 9e-6 floor, cosine decay
- TRAIN_SECONDS=7200 (estimated 50k+ steps)

## Key Directives

### Data Loading (CRITICAL CHANGE FROM v2)
1. **Load X_train for training** (not X_test)
2. **Use 80/20 train/val split on X_train** (v2 used 90/10 and trained on X_test)
3. **Load full X_test for final evaluation** (no holdout)

### Checkpointing (Persistent & Resumable)
1. **Save best model** on every improvement (metric: top1_structural_accuracy)
2. **Save latest model** every 1000 steps (for resumption)
3. **Save periodic snapshots** every 5000 steps (keep last 3)
4. **Checkpoints stored in:** `/Users/romain/experiments/glyco-dfm-v3/checkpoints/`

### Training Loop
1. Resume from `latest_model.pt` if it exists
2. Continue from saved step count on restart
3. Preserve optimizer state for momentum/adaptive updates

### Inference (Proper DFM)
1. Load best model at end of training
2. Implement Euler ODE solver: t=0 → t=1
3. Start with random tokens, iteratively refine via velocity field
4. Evaluate at N_STEPS ∈ [1, 4, 8, 16, 32]

### Evaluation Output
1. **Test set:** Full X_test (no holdout)
2. **Metric:** top1_structural_accuracy (via glycowork.compare_glycans)
3. **Pareto frontier:** accuracy vs N_steps table
4. **Speedup:** computed vs GlycoBART (207M, 96 steps)

## Success Criteria
- **N=1:** accuracy ≥ 0.8987 (one-shot, validates approach)
- **N=8-16:** accuracy ≥ 0.92 (target)
- Full X_test evaluation reported
- All checkpoints saved and resumable
- Pareto frontier saved to `models/pareto_frontier.json`

## Differences from v2
| Aspect | v2 | v3 |
|--------|----|----|
| Train data | X_test (90%) | X_train (80%) |
| Val data | X_test (10%) | X_train (20%) |
| Test data | None | X_test (full) |
| Data leakage | YES | NO |
| Train/val split | 90/10 | 80/10 (80/20 normalized) |
| Checkpointing | Minimal | Full (best, latest, periodic) |
| Resumable | NO | YES |
| Test evaluation | N/A | Pareto frontier |

## Status Indicators to Watch
```
FETCH_DONE X_train=XXX X_test=YYY       # Data loaded
SPLIT train=10000 val=2500              # Proper split confirmed
RESUME loading checkpoint                # Resumption working
CHECKPOINT best=X.XXXX step=YYYY         # Model improved
METRIC step=XXXX ... top1_structural_accuracy=X.XXXX  # Progress
PARETO_FRONTIER_SAVED                    # Final results computed
```
