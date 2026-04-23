import os
import math
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
DATA_DIR = "./dataset/public"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

# Model: decoder-only Transformer, ~4.88M params (under 50M limit)
D_MODEL  = 256
N_LAYERS = 6
N_HEADS  = 4
FFN_DIM  = 1024
MAX_LEN  = 512
DROPOUT  = 0.2

# Loss: 5x weight on continuation tokens, label smoothing for regularization
CONT_WEIGHT     = 5.0
LABEL_SMOOTHING = 0.1

# Composition-biased decoding: beta tuned on OOF (train-only, no test leakage)
COMP_BETA_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
COMP_EPS = 1e-6

# Training
N_FOLDS      = 5
EPOCHS       = 400
BATCH_SIZE   = 64
LR           = 1e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.05
GRAD_CLIP    = 1.0
PATIENCE     = 30
NUM_WORKERS  = 4

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP   = DEVICE.type == "cuda"
USE_BF16  = USE_AMP and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Vocabulary: 20 standard amino acids + 4 special tokens
AA_CHARS     = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
ALL_TOKS     = SPECIAL_TOKS + list(AA_CHARS)
VOCAB        = {t: i for i, t in enumerate(ALL_TOKS)}
ID2TOK       = {i: t for t, i in VOCAB.items()}
VOCAB_SIZE   = len(VOCAB)

PAD_ID = VOCAB["<PAD>"]
BOS_ID = VOCAB["<BOS>"]
EOS_ID = VOCAB["<EOS>"]
UNK_ID = VOCAB["<UNK>"]
AA_IDS = [VOCAB[aa] for aa in AA_CHARS]


def encode(seq):
    return [VOCAB.get(c, UNK_ID) for c in seq.upper()]


def build_composition_bias(context_str, beta):
    """Logit-space bias from context AA composition. Bayesian prior at inference."""
    if beta == 0.0:
        m = torch.full((VOCAB_SIZE,), float("-inf"), device=DEVICE)
        for i in AA_IDS:
            m[i] = 0.0
        return m
    counts = Counter(context_str)
    total = max(len(context_str), 1)
    m = torch.full((VOCAB_SIZE,), float("-inf"), device=DEVICE)
    for aa in AA_CHARS:
        freq = counts.get(aa, 0) / total + COMP_EPS
        m[VOCAB[aa]] = beta * math.log(freq)
    return m


class ProteinDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ctx_ids = encode(row["context"])
        if self.is_train:
            cont_ids = encode(row["continuation"])
            full = [BOS_ID] + ctx_ids + cont_ids
            if len(full) > MAX_LEN + 1:
                full = full[:MAX_LEN + 1]
            inp = full[:-1]
            tgt = full[1:]
            L = len(tgt)
            cont_start = len(ctx_ids)
            cont_mask = [False] * L
            for j in range(cont_start, min(cont_start + 20, L)):
                cont_mask[j] = True
            return inp, tgt, cont_mask
        else:
            return [BOS_ID] + ctx_ids, row["seq_id"]


def collate_train(batch):
    inps, tgts, cmasks = zip(*batch)
    B = len(inps)
    max_len = max(len(x) for x in inps)
    inp_t = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    tgt_t = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    cm_t = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (inp, tgt, cm) in enumerate(zip(inps, tgts, cmasks)):
        L = len(inp)
        inp_t[i, :L] = torch.tensor(inp, dtype=torch.long)
        tgt_t[i, :L] = torch.tensor(tgt, dtype=torch.long)
        cm_t[i, :L] = torch.tensor(cm, dtype=torch.bool)
    kp_mask = (inp_t == PAD_ID)
    return inp_t, tgt_t, cm_t, kp_mask


# Pre-norm causal Transformer block
class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True
        )
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, D_MODEL),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x, causal_mask, kp_mask):
        x2 = self.ln1(x)
        x2, _ = self.attn(
            x2, x2, x2,
            attn_mask=causal_mask,
            key_padding_mask=kp_mask,
            need_weights=False,
        )
        x = x + x2
        x = x + self.ff(self.ln2(x))
        return x


class ProteinGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([CausalBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, kp_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask=causal, kp_mask=kp_mask)
        return self.head(self.ln_f(x))


def cosine_schedule_with_warmup(optimizer, n_warmup, n_total):
    def lr_fn(step):
        if step < n_warmup:
            return float(step) / float(max(1, n_warmup))
        t = (step - n_warmup) / float(max(1, n_total - n_warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return max(LR_MIN / LR, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def weighted_cross_entropy(logits, targets, cont_mask, pad_id, label_smoothing):
    """CE with 5x weight on continuation tokens, label smoothing, PAD excluded."""
    V = logits.size(-1)
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)
    cont_flat = cont_mask.view(-1)
    valid = targets_flat != pad_id
    loss_per_token = F.cross_entropy(
        logits_flat, targets_flat,
        reduction="none",
        label_smoothing=label_smoothing,
        ignore_index=pad_id,
    )
    weights = torch.where(cont_flat, CONT_WEIGHT, 1.0)
    weights = weights * valid.float()
    return (loss_per_token * weights).sum() / weights.sum()


def train_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss, n_steps = 0.0, 0
    for inp, tgt, cm, kp_mask in loader:
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        cm = cm.to(DEVICE)
        kp_mask = kp_mask.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(inp, kp_mask=kp_mask)
            loss = weighted_cross_entropy(logits, tgt, cm, PAD_ID, LABEL_SMOOTHING)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(1, n_steps)


@torch.no_grad()
def eval_val_loss(model, loader):
    model.eval()
    total_loss, n_steps = 0.0, 0
    for inp, tgt, cm, kp_mask in loader:
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        cm = cm.to(DEVICE)
        kp_mask = kp_mask.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(inp, kp_mask=kp_mask)
            loss = weighted_cross_entropy(logits, tgt, cm, PAD_ID, LABEL_SMOOTHING)
        total_loss += loss.item()
        n_steps += 1
    return total_loss / max(1, n_steps)


@torch.no_grad()
def eval_continuation_accuracy(model, loader):
    """Teacher-forced per-residue accuracy on continuation tokens."""
    model.eval()
    correct = total = 0
    for inp, tgt, cm, kp_mask in loader:
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)
        cm = cm.to(DEVICE)
        kp_mask = kp_mask.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(inp, kp_mask=kp_mask)
        preds = logits.argmax(-1)
        correct += (preds[cm] == tgt[cm]).sum().item()
        total += cm.sum().item()
    return correct / max(1, total)


@torch.no_grad()
def generate_one(model, ctx_ids, comp_bias):
    """Greedy decode 20 tokens with composition bias added to logits."""
    model.eval()
    ids = torch.tensor([[BOS_ID] + ctx_ids], dtype=torch.long, device=DEVICE)
    generated = []
    for _ in range(20):
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(ids)
        nxt = (logits[0, -1] + comp_bias).argmax().item()
        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return "".join(ID2TOK[i] for i in generated)


@torch.no_grad()
def compute_oof_accuracy_beta(model, val_df, beta):
    """Free-running accuracy with composition bias at given beta."""
    model.eval()
    correct = total = 0
    for _, row in val_df.iterrows():
        comp_bias = build_composition_bias(row["context"], beta)
        pred = generate_one(model, encode(row["context"]), comp_bias)
        truth = row["continuation"]
        correct += sum(p == t for p, t in zip(pred, truth))
        total += 20
    return correct / total


def train_fold(fold_idx, tr_df, val_df):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx+1}/{N_FOLDS}  (train={len(tr_df)}, val={len(val_df)})")
    print(f"{'='*60}")
    sys.stdout.flush()

    tr_loader = DataLoader(
        ProteinDataset(tr_df, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_train, num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    )
    val_loader = DataLoader(
        ProteinDataset(val_df, is_train=True),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_train, num_workers=NUM_WORKERS, pin_memory=USE_AMP,
    )

    model = ProteinGPT().to(DEVICE)
    print(f"Parameters: {model.count_params()/1e6:.2f}M")
    assert model.count_params() < 50_000_000

    total_steps = EPOCHS * len(tr_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and not USE_BF16))

    best_val_loss = float("inf")
    ckpt_path = f"{WORK_DIR}/fold{fold_idx}_best.pt"
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, scaler)
        vl_loss = eval_val_loss(model, val_loader)
        tf_acc = eval_continuation_accuracy(model, val_loader)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), ckpt_path)
            patience = 0
            tag = "best"
        else:
            patience += 1
            tag = f"patience {patience}/{PATIENCE}"

        if epoch <= 10 or epoch % 10 == 0 or patience == 0 or patience >= PATIENCE:
            print(f"  ep {epoch:3d}  tr={tr_loss:.4f}  val={vl_loss:.4f}  tf_acc={tf_acc:.4f}  [{tag}]")
            sys.stdout.flush()

        if patience >= PATIENCE:
            print(f"  Early stop at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


def main():
    print(f"Device: {DEVICE}  AMP: {USE_AMP}  dtype: {AMP_DTYPE}")
    print(f"Model: Transformer d={D_MODEL} L={N_LAYERS} H={N_HEADS}")
    print(f"Loss: cont_weight={CONT_WEIGHT} label_smoothing={LABEL_SMOOTHING}")
    print(f"Training: LR={LR} dropout={DROPOUT} wd={WEIGHT_DECAY} epochs={EPOCHS}")

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    # Phase 1: Train 5-fold models
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_models, fold_val_dfs = [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        model = train_fold(fold_idx, train_df.iloc[tr_idx], train_df.iloc[val_idx])
        fold_models.append(model)
        fold_val_dfs.append(train_df.iloc[val_idx])

    # Phase 2: Tune composition bias beta on OOF
    print(f"\n{'='*60}")
    print("Tuning composition beta on OOF...")
    print(f"{'='*60}")
    sys.stdout.flush()

    beta_scores = {}
    for beta in COMP_BETA_CANDIDATES:
        fold_accs = []
        for model, val_df in zip(fold_models, fold_val_dfs):
            acc = compute_oof_accuracy_beta(model, val_df, beta)
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        beta_scores[beta] = mean_acc
        print(f"  beta={beta:.1f}  OOF={mean_acc:.4f}  folds={[f'{a:.4f}' for a in fold_accs]}")
        sys.stdout.flush()

    best_beta = max(beta_scores, key=beta_scores.get)
    print(f"\nBest beta: {best_beta}  OOF: {beta_scores[best_beta]:.4f}")
    print(f"Baseline (beta=0): {beta_scores[0.0]:.4f}")
    sys.stdout.flush()

    # Phase 3: Collect OOF predictions with best beta
    oof_rows = []
    fold_accs_final = []
    for model, val_df in zip(fold_models, fold_val_dfs):
        correct = total = 0
        for _, row in val_df.iterrows():
            comp_bias = build_composition_bias(row["context"], best_beta)
            pred = generate_one(model, encode(row["context"]), comp_bias)
            truth = row["continuation"]
            sc = sum(p == t for p, t in zip(pred, truth))
            correct += sc
            total += 20
            oof_rows.append({
                "seq_id": row["seq_id"],
                "pred": pred,
                "truth": truth,
                "accuracy": sc / 20,
            })
        fold_accs_final.append(correct / total)

    print(f"\n{'='*60}")
    print(f"OOF per fold: {[f'{a:.4f}' for a in fold_accs_final]}")
    print(f"Mean OOF: {np.mean(fold_accs_final):.4f} +/- {np.std(fold_accs_final):.4f}")
    print(f"{'='*60}")

    pd.DataFrame(oof_rows).to_csv(f"{WORK_DIR}/oof_predictions.csv", index=False)

    # Phase 4: Generate test predictions with 5-fold ensemble + best beta
    print(f"\nGenerating test predictions (5-fold ensemble, beta={best_beta})...")
    continuations = []

    for _, row in test_df.iterrows():
        ctx_str = row["context"]
        ctx_ids = encode(ctx_str)
        comp_bias = build_composition_bias(ctx_str, best_beta)
        base_seq = [BOS_ID] + ctx_ids

        running = [
            torch.tensor([base_seq], dtype=torch.long, device=DEVICE)
            for _ in fold_models
        ]
        generated = []

        for _ in range(20):
            avg_logits = None
            for model, seq in zip(fold_models, running):
                model.eval()
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP
                ):
                    lg = model(seq)[0, -1]
                avg_logits = lg if avg_logits is None else avg_logits + lg

            avg_logits = avg_logits / len(fold_models) + comp_bias
            nxt = avg_logits.argmax().item()
            generated.append(nxt)

            nxt_t = torch.tensor([[nxt]], device=DEVICE)
            running = [torch.cat([s, nxt_t], dim=1) for s in running]

        continuations.append("".join(ID2TOK[i] for i in generated))

    sub = pd.DataFrame({"seq_id": test_df["seq_id"], "continuation": continuations})

    assert len(sub) == len(test_df)
    assert (sub["continuation"].str.len() == 20).all()
    assert sub["continuation"].apply(lambda x: all(c in set(AA_CHARS) for c in x)).all()

    sub.to_csv(f"{WORK_DIR}/submission.csv", index=False)
    print(f"Saved: {WORK_DIR}/submission.csv")
    print(sub.head(10).to_string(index=False))
    print(f"\nFinal: beta={best_beta}  CV={np.mean(fold_accs_final):.4f}")


if __name__ == "__main__":
    main()
