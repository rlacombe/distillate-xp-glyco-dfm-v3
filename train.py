# requirements: requests glycowork
"""
Glyco DFM v3: Discrete Flow Matching for Glycan Structure Prediction

Data: CandyCrush v2 (Zenodo)
- X_train.pkl → training set (80/20 train/val split)
- X_test.pkl → test set (full evaluation)
- Baseline: GlycoBART 0.8903 on same test set

Algorithm: Proper DFM with ODE solver (Gat et al., NeurIPS 2024)
- Training: continuous t ∈ [0,1], standard masked corruption
- Inference: Euler ODE solver with flexible step counts
- Evaluation: Pareto frontier (accuracy vs N_steps=[1,4,8,16,32])

Checkpointing: Persistent to experiment directory
- best_model.pt: best by top1_structural_accuracy
- latest_model.pt: for resumption
- checkpoint_step_XXXXXX.pt: periodic snapshots
- models/final_model.pt: final checkpoint
"""
import os, time, math, pickle, re, json
from pathlib import Path
from datetime import datetime
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if os.getenv("DISTILLATE_COMPUTE") == "hfjobs":
    CHECKPOINT_DIR = Path("/output/checkpoints")
    MODELS_DIR = Path("/output/models")
else:
    CHECKPOINT_DIR = Path("/Users/romain/experiments/glyco-dfm-v3/checkpoints")
    MODELS_DIR = Path("/Users/romain/experiments/glyco-dfm-v3/models")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ZENODO = "https://zenodo.org/records/10997110/files"

def fetch(name, url, cache=None):
    if cache is None:
        if os.getenv("DISTILLATE_COMPUTE") == "hfjobs":
            # Prefer /data (HF dataset mount) if it exists; else fall back to /tmp
            data_dir = Path("/data")
            cache = data_dir if data_dir.is_dir() else Path("/tmp")
        else:
            cache = Path("/tmp")

    cache.mkdir(parents=True, exist_ok=True)
    p = cache / name
    if not p.exists():
        print(f"DOWNLOAD {name} <- {url}", flush=True)
        t0 = time.time()
        r = requests.get(url, stream=True, timeout=1800)
        r.raise_for_status()
        with open(p, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk: f.write(chunk)
        sz = p.stat().st_size / 1e6
        print(f"DOWNLOAD_OK {name} size={sz:.1f}MB dt={time.time()-t0:.1f}s", flush=True)
    else:
        print(f"CACHE_HIT {name} from {cache}", flush=True)

    print(f"UNPICKLE {name}...", flush=True); t0 = time.time()
    obj = pickle.loads(p.read_bytes())
    print(f"UNPICKLE_OK {name} dt={time.time()-t0:.1f}s", flush=True)
    return obj

T0 = time.time()
# Load training set
X_tr = fetch("X_train.pkl", f"{ZENODO}/X_train_CC2_240110.pkl?download=1")
y_tr = fetch("y_train.pkl", f"{ZENODO}/y_train_CC2_240110.pkl?download=1")

# Load test set (for final evaluation)
X_te = fetch("X_test.pkl",  f"{ZENODO}/X_test_CC2_240110.pkl?download=1")
y_te = fetch("y_test.pkl",  f"{ZENODO}/y_test_CC2_240110.pkl?download=1")

print(f"FETCH_DONE X_train={len(X_tr)} X_test={len(X_te)} total_dt={time.time()-T0:.1f}s", flush=True)

# Check glycowork availability for structural comparison
try:
    from glycowork.motif.graph import compare_glycans
    HAS_GLYCOWORK = True
    print("GLYCOWORK_OK", flush=True)
except ImportError:
    HAS_GLYCOWORK = False
    print("GLYCOWORK_MISSING: falling back to string comparison", flush=True)

def struct_equal(a, b):
    """Return True if two IUPAC-condensed glycan strings are structurally equivalent."""
    if a == b:
        return True
    if not HAS_GLYCOWORK:
        return False
    try:
        return compare_glycans(a, b)
    except Exception:
        return False

def to_list(y):
    if isinstance(y, list): return y
    if isinstance(y, tuple): return list(y)
    if hasattr(y, "tolist"): return y.tolist()
    return list(y)

# Use X_train (not X_test) for training
X_all = to_list(X_tr)
y_all = to_list(y_tr)
print(f"N total train={len(X_all)}", flush=True)

def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)

# Filter to non-empty samples
y_strs = [as_str(y) for y in y_all]
keep = [i for i, s in enumerate(y_strs) if s]
rng = np.random.default_rng(42)
rng.shuffle(keep)

# 80/20 train/val split (match GlycoBART, different from v2's 90/10)
split = int(0.80 * len(keep))
train_idx = keep[:split]
val_idx = keep[split:]

print(f"SPLIT train={len(train_idx)} val={len(val_idx)}", flush=True)

def get_feats(x):
    mz   = np.asarray(x[0], dtype=np.float32).flatten()
    mz_r = np.asarray(x[1], dtype=np.float32).flatten() if len(x) > 1 else np.zeros_like(mz)
    try: prec = float(x[2]) if len(x) > 2 else 0.0
    except (TypeError, ValueError): prec = 0.0
    return mz, mz_r, prec

MZ_LEN = get_feats(X_all[0])[0].shape[0]

def stack_feats(idx, source=None):
    if source is None:
        source = X_all
    N = len(idx)
    MZ   = np.zeros((N, MZ_LEN), dtype=np.float32)
    MZR  = np.zeros((N, MZ_LEN), dtype=np.float32)
    PREC = np.zeros((N,),         dtype=np.float32)
    for i, j in enumerate(idx):
        mz, mz_r, prec = get_feats(source[j])
        n = min(mz.shape[0], MZ_LEN); MZ[i, :n] = mz[:n]
        n = min(mz_r.shape[0], MZ_LEN); MZR[i, :n] = mz_r[:n]
        PREC[i] = prec
    return MZ, MZR, PREC

MZ_tr, MZR_tr, PREC_tr = stack_feats(train_idx)
MZ_va, MZR_va, PREC_va = stack_feats(val_idx)
y_tr_kept = [y_strs[j] for j in train_idx]
y_va_kept = [y_strs[j] for j in val_idx]

TOKEN_RE = re.compile(r"([\(\)\[\]]|a\d-\d|b\d-\d|a\?-\?|b\?-\?|a\d-\?|b\d-\?)")
def tokenize(s): return [p for p in TOKEN_RE.split(s) if p]

PAD, MASK, BOS, EOS = "<pad>", "<mask>", "<bos>", "<eos>"
specials = [PAD, MASK, BOS, EOS]
vocab = {t: i for i, t in enumerate(specials)}
for s in y_tr_kept:
    for t in tokenize(s):
        if t not in vocab: vocab[t] = len(vocab)
V = len(vocab); PAD_ID, MASK_ID, BOS_ID, EOS_ID = vocab[PAD], vocab[MASK], vocab[BOS], vocab[EOS]
id2tok = {v: k for k, v in vocab.items()}
print(f"VOCAB V={V}", flush=True)

MAX_LEN = 96
def encode(s):
    ids = [BOS_ID] + [vocab.get(t, MASK_ID) for t in tokenize(s)][:MAX_LEN-2] + [EOS_ID]
    return ids + [PAD_ID]*(MAX_LEN-len(ids))

def decode_ids(ids):
    """Convert token ids back to IUPAC string (strip BOS/EOS/PAD)."""
    tokens = []
    for i in ids:
        if i in (BOS_ID, PAD_ID): continue
        if i == EOS_ID: break
        tokens.append(id2tok.get(i, "?"))
    return "".join(tokens)

Y_tr = torch.tensor([encode(s) for s in y_tr_kept], dtype=torch.long)
Y_va = torch.tensor([encode(s) for s in y_va_kept], dtype=torch.long)

def norm_spec(x):
    x = torch.log1p(x.clamp_min(0))
    return x / (x.max(-1, keepdim=True).values + 1e-6)

MZ_tr_t  = norm_spec(torch.tensor(MZ_tr));   MZR_tr_t = norm_spec(torch.tensor(MZR_tr))
MZ_va_t  = norm_spec(torch.tensor(MZ_va));   MZR_va_t = norm_spec(torch.tensor(MZR_va))
pt = torch.tensor(PREC_tr); pm, ps = pt.mean(), pt.std().clamp_min(1.0)
PREC_tr_t = (pt - pm) / ps
PREC_va_t = (torch.tensor(PREC_va) - pm) / ps
print(f"TENSORS Y_tr={tuple(Y_tr.shape)} Y_va={tuple(Y_va.shape)}", flush=True)

# 50M model — d=512, 4L enc, 8L dec, 8 heads
D_MODEL = 512; N_HEADS = 8; N_ENC = 4; N_DEC = 8; N_SPEC_TOKENS = 64

class SpecEncoder(nn.Module):
    def __init__(self, mz_len, d=D_MODEL, n_layers=N_ENC, n_heads=N_HEADS, n_tokens=N_SPEC_TOKENS):
        super().__init__()
        self.mz_len = mz_len; self.n_tokens = n_tokens
        self.chunk = mz_len // n_tokens + (1 if mz_len % n_tokens else 0)
        self.proj = nn.Linear(2 * self.chunk, d)
        self.pos  = nn.Embedding(n_tokens + 1, d)
        self.prec_proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                          dropout=0.1, batch_first=True, activation="gelu",
                                          norm_first=True)
        self.trunk = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.register_buffer("ids", torch.arange(n_tokens + 1).unsqueeze(0))

    def forward(self, mz, mz_r, prec):
        B = mz.shape[0]
        pad = self.chunk * self.n_tokens - self.mz_len
        mz_p  = F.pad(mz,  (0, pad)).view(B, self.n_tokens, self.chunk)
        mzr_p = F.pad(mz_r,(0, pad)).view(B, self.n_tokens, self.chunk)
        h = self.proj(torch.cat([mz_p, mzr_p], dim=-1))
        p = self.prec_proj(prec.view(B, 1, 1)).expand(B, 1, -1)
        h = torch.cat([h, p], dim=1) + self.pos(self.ids.expand(B, -1))
        return self.trunk(h)

class CondDFM(nn.Module):
    def __init__(self, V, d=D_MODEL, n_layers=N_DEC, n_heads=N_HEADS, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(max_len, d)
        self.time_emb = nn.Linear(1, d)
        dec = nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                          dropout=0.1, batch_first=True, activation="gelu",
                                          norm_first=True)
        self.trunk = nn.TransformerDecoder(dec, num_layers=n_layers)
        self.head = nn.Linear(d, V)
        self.register_buffer("ids", torch.arange(max_len).unsqueeze(0))

    def forward(self, x, t, mem):
        B, L = x.shape
        h = self.emb(x) + self.pos(self.ids[:, :L])
        h = h + self.time_emb(t.view(B, 1, 1).float()).expand(-1, L, -1)
        return self.head(self.trunk(h, mem, tgt_key_padding_mask=(x == PAD_ID)))

enc = SpecEncoder(MZ_LEN).to(DEVICE)
dec = CondDFM(V).to(DEVICE)
n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
print(f"MODEL n_params={n_params}", flush=True)

TRAIN_SECONDS = int(os.environ.get("TRAIN_SECONDS", 7200))
# Estimated ~5.88 steps/s on L40S for 50M params
TOTAL_STEPS_EST = int(5.88 * TRAIN_SECONDS)
print(f"SCHEDULE TOTAL_STEPS_EST={TOTAL_STEPS_EST} TRAIN_SECONDS={TRAIN_SECONDS}", flush=True)
WARMUP_STEPS = 200
LR_MAX = 3e-4
opt = torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()), lr=LR_MAX, weight_decay=0.01)

# LR schedule: cosine decay with 0.03×LR_MAX floor
def get_lr(step):
    if step < WARMUP_STEPS:
        return LR_MAX * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS_EST - WARMUP_STEPS, 1)
    return LR_MAX * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

def corrupt(x):
    t = torch.rand(x.shape[0], device=x.device)
    non_special = (x != PAD_ID) & (x != BOS_ID) & (x != EOS_ID)
    mask = (torch.rand_like(x, dtype=torch.float) < t.unsqueeze(1)) & non_special
    return torch.where(mask, torch.full_like(x, MASK_ID), x), mask, t

BATCH = 128

@torch.no_grad()
def evaluate():
    enc.eval(); dec.eval()
    tc=0; tt=0; tl=0.0
    em_struct_os=0; n_em=0

    for i in range(0, len(Y_va), BATCH):
        yb  = Y_va[i:i+BATCH].to(DEVICE)
        mb  = MZ_va_t[i:i+BATCH].to(DEVICE)
        mrb = MZR_va_t[i:i+BATCH].to(DEVICE)
        pb  = PREC_va_t[i:i+BATCH].to(DEVICE)
        mem = enc(mb, mrb, pb)

        # Token-level masked loss
        xc, m, t = corrupt(yb)
        logits = dec(xc, t, mem)
        if m.sum().item() > 0:
            tl += F.cross_entropy(logits[m], yb[m]).item() * m.sum().item()
            tc += (logits.argmax(-1)[m] == yb[m]).sum().item()
            tt += m.sum().item()

        non_spec = (yb != PAD_ID) & (yb != BOS_ID) & (yb != EOS_ID)

        # One-shot decode (single forward pass at t=1, all-masked)
        xm = torch.where(non_spec, torch.full_like(yb, MASK_ID), yb)
        tones = torch.ones(xm.size(0), device=DEVICE)
        p1 = dec(xm, tones, mem).argmax(-1)

        # p1_clean restores BOS/EOS/PAD from ground truth so decode_ids gives a clean string
        p1_clean = torch.where(non_spec, p1, yb)
        if HAS_GLYCOWORK:
            for b in range(yb.size(0)):
                if struct_equal(decode_ids(p1_clean[b].tolist()), decode_ids(yb[b].tolist())):
                    em_struct_os += 1
        else:
            em_struct_os += ((p1 == yb) | ~non_spec).all(dim=1).sum().item()

        n_em += yb.size(0)

    enc.train(); dec.train()
    return (tc/max(tt,1), math.exp(tl/max(tt,1)), em_struct_os/max(n_em,1))

# ============================================================================
# CheckpointManager class
# ============================================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_count=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_count = keep_count
        self.best_metric = 0.0

    def save_best(self, checkpoint_dict, metric_value, step):
        """Save best model checkpoint."""
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            torch.save(checkpoint_dict, self.checkpoint_dir / "best_model.pt")
            print(f"CHECKPOINT best={metric_value:.4f} step={step}", flush=True)
            return True
        return False

    def save_latest(self, checkpoint_dict, step):
        """Save latest checkpoint for resumption."""
        torch.save(checkpoint_dict, self.checkpoint_dir / "latest_model.pt")

    def save_periodic(self, checkpoint_dict, step):
        """Save periodic checkpoint and cleanup old ones."""
        path = self.checkpoint_dir / f"checkpoint_step_{step:06d}.pt"
        torch.save(checkpoint_dict, path)
        self.cleanup_old()

    def cleanup_old(self):
        """Keep only most recent N checkpoints."""
        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        for old in ckpts[:-self.keep_count]:
            old.unlink()

    def load_latest(self):
        """Load latest checkpoint if it exists."""
        path = self.checkpoint_dir / "latest_model.pt"
        if path.exists():
            return torch.load(path, map_location=DEVICE)
        return None

# Initialize checkpoint manager
ckpt_mgr = CheckpointManager(CHECKPOINT_DIR, keep_count=3)

# Load latest checkpoint if exists
checkpoint = ckpt_mgr.load_latest()
if checkpoint:
    print(f"RESUME loading checkpoint from step {checkpoint['step']}", flush=True)
    enc.load_state_dict(checkpoint["encoder_state"])
    dec.load_state_dict(checkpoint["decoder_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"]
    best_struct = checkpoint.get("best_struct", 0.0)
    print(f"RESUME restored: step={start_step} best_struct={best_struct:.4f}", flush=True)
else:
    start_step = 0
    best_struct = 0.0

print(f"SCHEDULE TOTAL_STEPS_EST={TOTAL_STEPS_EST} start_step={start_step}", flush=True)

print(f"TRAIN_START budget={TRAIN_SECONDS}s n_train={len(Y_tr)} n_val={len(Y_va)}", flush=True)
t_start = time.time(); step = start_step
best_acc = 0.0
EVAL_FREQ = 500

while time.time() - t_start < TRAIN_SECONDS:
    lr = get_lr(step)
    for pg in opt.param_groups: pg["lr"] = lr

    idx = torch.randint(0, len(Y_tr), (BATCH,))
    yb  = Y_tr[idx].to(DEVICE)
    mem = enc(MZ_tr_t[idx].to(DEVICE), MZR_tr_t[idx].to(DEVICE), PREC_tr_t[idx].to(DEVICE))
    xc, m, t = corrupt(yb)
    if m.sum().item() == 0: step += 1; continue
    loss = F.cross_entropy(dec(xc, t, mem)[m], yb[m])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 1.0)
    opt.step(); step += 1

    if step % EVAL_FREQ == 0:
        acc, ppl, struct = evaluate()
        print(f"METRIC step={step} lr={lr:.2e} train_loss={loss.item():.4f} "
              f"val_token_acc={acc:.4f} val_ppl={ppl:.3f} "
              f"top1_structural_accuracy={struct:.4f} "
              f"elapsed={time.time()-t_start:.1f}", flush=True)

        if struct > best_struct:
            best_struct = struct
        if acc > best_acc: best_acc = acc

        # Save checkpoints
        full_checkpoint = {
            "encoder_state": enc.state_dict(),
            "decoder_state": dec.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": step,
            "best_struct": best_struct,
            "model_config": {
                "d_model": D_MODEL,
                "n_enc": N_ENC,
                "n_dec": N_DEC,
                "vocab_size": V,
                "max_len": MAX_LEN,
                "mz_len": MZ_LEN,
            }
        }

        # Save best model on improvement
        ckpt_mgr.save_best(full_checkpoint, struct, step)

        # Save latest for resumption
        ckpt_mgr.save_latest(full_checkpoint, step)

        # Save periodic snapshot every 5000 steps
        if step % 5000 == 0 and step > 0:
            ckpt_mgr.save_periodic(full_checkpoint, step)

acc, ppl, struct = evaluate()
print(f"METRIC_FINAL step={step} val_token_acc={acc:.4f} val_ppl={ppl:.3f} "
      f"top1_structural_accuracy={struct:.4f} "
      f"best_val_token_acc={best_acc:.4f} "
      f"best_top1_structural_accuracy={best_struct:.4f}", flush=True)
print(f"DONE elapsed={time.time()-t_start:.1f}s", flush=True)

# ============================================================================
# DFM Multi-step Evaluation on Full Test Set
# ============================================================================
print(f"\n=== PARETO FRONTIER: DFM Step Count vs Accuracy ===", flush=True)
print(f"N_STEPS | Top1_Structural_Accuracy | Speedup vs GlycoBART", flush=True)
print("-" * 70, flush=True)

# Load best model
best_ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE)
enc.load_state_dict(best_ckpt["encoder_state"])
dec.load_state_dict(best_ckpt["decoder_state"])

# Prepare test set
X_test_all = to_list(X_te)
y_test_all = to_list(y_te)
y_test_strs = [as_str(y) for y in y_test_all]
test_idx = [i for i, s in enumerate(y_test_strs) if s]

print(f"TEST_SET: {len(test_idx)} samples", flush=True)

MZ_te, MZR_te, PREC_te = stack_feats(test_idx, source=X_test_all)
y_test_kept = [y_test_strs[i] for i in test_idx]

MZ_te_t = norm_spec(torch.tensor(MZ_te))
MZR_te_t = norm_spec(torch.tensor(MZR_te))
PREC_te_t = (torch.tensor(PREC_te) - pm) / ps

def evaluate_dfm_n_steps(n_steps):
    """Evaluate with proper DFM ODE solver at n_steps."""
    enc.eval(); dec.eval()
    correct = 0; total = 0
    dt = 1.0 / n_steps

    for i in range(0, len(y_test_kept), BATCH):
        batch_idx = slice(i, min(i+BATCH, len(y_test_kept)))
        yb = torch.tensor([encode(s) for s in y_test_kept[batch_idx]],
                         dtype=torch.long).to(DEVICE)
        mb = MZ_te_t[batch_idx].to(DEVICE)
        mrb = MZR_te_t[batch_idx].to(DEVICE)
        pb = PREC_te_t[batch_idx].to(DEVICE)

        # Encode spectrum (constant)
        mem = enc(mb, mrb, pb)

        # Initialize: random tokens at t=0
        batch_size = yb.shape[0]
        x_t = torch.randint(0, V, (batch_size, MAX_LEN), device=DEVICE)

        # ODE Euler solver: t=0 → t=1
        t = 0.0
        while t < 1.0 - 1e-3:
            # Model predicts target distribution p_1
            t_tensor = torch.full((batch_size,), t, device=DEVICE)
            logits = dec(x_t, t_tensor, mem)
            p_1 = torch.softmax(logits, dim=-1)

            # Current state as one-hot
            x_one_hot = torch.nn.functional.one_hot(x_t, V).float()

            # Velocity field: direction towards target
            u = (p_1 - x_one_hot) / max(1.0 - t, 1e-5)

            # Adaptive step
            h = min(dt, 1.0 - t)

            # Euler update
            new_probs = x_one_hot + h * u
            new_probs = torch.clamp(new_probs, 0.0, 1.0)
            new_probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Sample next tokens
            x_t = torch.distributions.Categorical(probs=new_probs).sample()

            t += h

        # Structural accuracy
        for j, (pred, truth) in enumerate(zip(x_t, yb)):
            pred_str = decode_ids(pred.cpu().numpy())
            truth_str = decode_ids(truth.cpu().numpy())
            if struct_equal(pred_str, truth_str):
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0

# Evaluate at multiple step counts
n_steps_list = [1, 4, 8, 16, 32]
results = {}

for n_steps in n_steps_list:
    acc = evaluate_dfm_n_steps(n_steps)
    results[n_steps] = acc

    # Speedup estimate vs GlycoBART (207M autoregressive, ~96 passes)
    # GlycoBART: 207M params, 96 autoregressive steps
    # DFM: 50M params, n_steps deterministic steps
    speedup = (96 * 207) / (n_steps * 50)

    print(f"{n_steps:7d} | {acc:25.4f} | {speedup:18.1f}x", flush=True)

# Save results
results_json = {
    "algorithm": "Discrete Flow Matching",
    "model_size": "50M parameters",
    "data": "CandyCrush v2",
    "results": {str(n): acc for n, acc in results.items()},
    "baseline_glycobart": 0.8903,
    "test_set_size": len(test_idx),
    "timestamp": datetime.now().isoformat(),
}

with open(MODELS_DIR / "pareto_frontier.json", "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\nPARETOFRONTIER_SAVED to {MODELS_DIR / 'pareto_frontier.json'}", flush=True)

# Export final model
def export_inference_model(checkpoint_path, export_path):
    """Export best model for inference."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    inference_ckpt = {
        "encoder_state": ckpt["encoder_state"],
        "decoder_state": ckpt["decoder_state"],
        "vocab": vocab,
        "model_config": ckpt.get("model_config", {}),
    }

    torch.save(inference_ckpt, export_path)
    print(f"EXPORT inference model to {export_path}", flush=True)

export_inference_model(
    CHECKPOINT_DIR / "best_model.pt",
    MODELS_DIR / "best_model_inference.pt"
)

print(f"TRAINING_COMPLETE", flush=True)
