# Steering: Data Download Optimization via HF Dataset Mount

## Objective
Eliminate repetitive data downloads (14GB+ per job) by uploading CandyCrush v2 to a private HF dataset and mounting it read-only in HF Jobs.

## Status
- ✅ **compute_hfjobs.py bug fixed:** Volume parsing now uses `rsplit(":", 1)` to preserve `hf://` URLs
- ✅ **train.py updated:** fetch() now checks `/data` first (on HF Jobs) before downloading
- ✅ **hf_jobs.json configured:** `volumes: ["hf://datasets/rlacombe/glyco-candycrush-v2:/data"]`

## Next Steps (One-Time Setup)

### 1. Create HF Dataset Repository
```bash
huggingface-cli repo create glyco-candycrush-v2 --type dataset --private
```

### 2. Upload the 4 pickle files
Download once locally (or use existing cache), then push to HF:

```bash
cd /tmp
git lfs install
git clone https://huggingface.co/datasets/rlacombe/glyco-candycrush-v2
cd glyco-candycrush-v2

# Download if needed (or copy from existing cache)
# X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl

git add *.pkl
git commit -m "Initial: CandyCrush v2 dataset"
git push
```

### 3. Test on HF Jobs
Run a short test job with 60s timeout:
```bash
TRAIN_SECONDS=60 python train.py
# Should see "CACHE_HIT X_train.pkl from /data" if mounted correctly
```

## Expected Benefits

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| **Data download time** | ~120s per job | 0s | **120s per job** |
| **Total job time** | ~7500s | ~7380s | **1.6% speedup** |
| **Network usage** | 14GB per job | 0 | **14GB per job** |
| **Cost per run** | ~$5.20 | ~$5.00 | **$0.20 per run** |

## References

- **compute_hfjobs.py fix:** Line 232-235, use `rsplit` for volume parsing
- **train.py fetch():** Lines 44-70, DISTILLATE_COMPUTE check
- **hf_jobs.json:** volumes array with dataset mount spec

## Backoff Plan

If HF dataset upload fails:
1. Keep using temporary /tmp downloads (current behavior)
2. Or mount a persistent Hugging Face Bucket at /output/data (Option 1 fallback)

## Notes

- This is a one-time setup cost (~5 min) for unlimited future savings
- Subsequent runs on HF Jobs will have zero download time
- Read-only mount prevents accidental writes to the dataset
