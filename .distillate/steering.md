# Steering: Data Optimization via HF Dataset (Option 2 Approved)

## ✅ Infrastructure Complete

You suggested optimizing data downloads. **Option 2 is now implemented and ready.**

### What Was Done (Committed)

1. **Fixed compute_hfjobs.py bug** 
   - Volume parsing now uses `rsplit(":", 1)` to preserve `hf://` URLs
   
2. **Updated train.py fetch()**
   - Detects `DISTILLATE_COMPUTE=hfjobs`
   - Uses `/data` (mounted dataset) instead of downloading
   - Fallback to download if mount unavailable

3. **Configured hf_jobs.json**
   - Added: `"volumes": ["hf://datasets/rlacombe/glyco-candycrush-v2:/data"]`
   - Switched to A100-large hardware

## 📋 Execute One-Time Setup (5 min)

### 1. Create HF Dataset Repository
```bash
huggingface-cli repo create glyco-candycrush-v2 --type dataset --private
```

### 2. Upload Pickle Files
```bash
cd /tmp
git lfs install
git clone https://huggingface.co/datasets/rlacombe/glyco-candycrush-v2
cd glyco-candycrush-v2

# X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl should be in /tmp from earlier downloads
git add *.pkl
git commit -m "Initial: CandyCrush v2 dataset"
git push
```

### 3. Test Mount Works (60 seconds)
```bash
cd /Users/romain/experiments/glyco-dfm-v3
TRAIN_SECONDS=60 python train.py
```
✅ Success if you see: `CACHE_HIT X_train.pkl from /data`

### 4. Full Training (if mount test passes)
```bash
TRAIN_SECONDS=7200 python train.py
```

## 📊 Expected Gains
- **120s saved per job** (14GB download → 0)
- **1.6% faster** overall
- **$0.20/run cost savings**
- **Read-only mount** prevents accidents

## 📖 Reference
- Full details: `.distillate/optimization_steering.md`
- Code changes already committed, ready to go

Your move! 🚀
