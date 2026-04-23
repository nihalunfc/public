#!/usr/bin/env python3
"""
Protein Sequence Continuation Prediction - V2
Improved ensemble: Multiple encoders + Causal LM + N-gram
Better augmentation, regularization, and position-aware ensembling.
"""

import os, math, random, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

# â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONT_LEN = 20
MAX_LEN = 400

AA_LIST = list('ACDEFGHIKLMNPQRSTUVWY')
NUM_AA = len(AA_LIST)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
VOCAB_SIZE = NUM_AA + 1  # 0=pad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, 'dataset', 'public')):
    DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'public')
else:
    DATA_DIR = os.path.join(BASE_DIR, 'public')
WORKING_DIR = os.path.join(BASE_DIR, 'working')
os.makedirs(WORKING_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# â”€â”€â”€ N-gram Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class NgramModel:
    def __init__(self):
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.fourgram = defaultdict(Counter)
        self.unigram = Counter()
        self.total = 0

    def fit(self, sequences):
        for seq in sequences:
            for i, c in enumerate(seq):
                self.unigram[c] += 1
                self.total += 1
                if i >= 1: self.bigram[seq[i-1]][c] += 1
                if i >= 2: self.trigram[seq[i-2:i]][c] += 1
                if i >= 3: self.fourgram[seq[i-3:i]][c] += 1

    def probs(self, prefix):
        p = np.array([(self.unigram.get(aa, 0) + 1) for aa in AA_LIST], dtype=np.float64)
        p /= p.sum()

        if len(prefix) >= 1:
            key = prefix[-1]
            if key in self.bigram:
                n = sum(self.bigram[key].values())
                if n >= 2:
                    q = np.array([(self.bigram[key].get(aa, 0) + 0.01) for aa in AA_LIST])
                    q /= q.sum()
                    w = n / (n + 3.0)
                    p = (1 - w) * p + w * q

        if len(prefix) >= 2:
            key = prefix[-2:]
            if key in self.trigram:
                n = sum(self.trigram[key].values())
                if n >= 3:
                    q = np.array([(self.trigram[key].get(aa, 0) + 0.01) for aa in AA_LIST])
                    q /= q.sum()
                    w = n / (n + 5.0)
                    p = (1 - w) * p + w * q

        if len(prefix) >= 3:
            key = prefix[-3:]
            if key in self.fourgram:
                n = sum(self.fourgram[key].values())
                if n >= 5:
                    q = np.array([(self.fourgram[key].get(aa, 0) + 0.01) for aa in AA_LIST])
                    q /= q.sum()
                    w = n / (n + 8.0)
                    p = (1 - w) * p + w * q

        return p / p.sum()

    def continuation_probs(self, context):
        out = np.zeros((CONT_LEN, NUM_AA))
        cur = context
        for i in range(CONT_LEN):
            out[i] = self.probs(cur)
            cur += AA_LIST[np.argmax(out[i])]
        return out


# â”€â”€â”€ Encoder Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def tok(seq):
    return [AA_TO_IDX[c] + 1 for c in seq]


class Encoder(nn.Module):
    def __init__(self, d=192, heads=6, layers=4, ff=768, drop=0.25):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(CONT_LEN, d) * 0.02)
        self.ca = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.cln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(ff, NUM_AA))
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'embed' not in n:
                nn.init.xavier_uniform_(p)
        print(f"  Encoder: {sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, x, mask=None):
        B, L = x.shape
        h = self.embed(x) * math.sqrt(self.d) + self.pos(torch.arange(L, device=x.device))
        h = self.drop(h)
        h = self.ln(self.enc(h, src_key_padding_mask=mask))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        o, _ = self.ca(q, h, h, key_padding_mask=mask)
        return self.head(self.cln(o + q))


# â”€â”€â”€ Causal LM Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class CausalLM(nn.Module):
    def __init__(self, d=192, heads=6, layers=4, ff=768, drop=0.25):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(VOCAB_SIZE, d, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN + CONT_LEN + 1, d)
        self.drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE)
        for n, p in self.named_parameters():
            if p.dim() > 1 and 'embed' not in n:
                nn.init.xavier_uniform_(p)
        print(f"  CausalLM: {sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, x):
        B, L = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.embed(x) * math.sqrt(self.d) + self.pos(torch.arange(L, device=x.device))
        h = self.drop(h)
        h = self.ln(self.enc(h, mask=mask, is_causal=True))
        return self.head(h)

    @torch.no_grad()
    def get_continuation_probs(self, context_ids, length=CONT_LEN):
        self.eval()
        ids = context_ids.clone()
        all_probs = []
        for _ in range(length):
            if ids.size(1) > MAX_LEN:
                inp = ids[:, -MAX_LEN:]
            else:
                inp = ids
            logits = self(inp)
            next_logits = logits[:, -1, :]
            next_logits[:, 0] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            aa_probs = torch.stack([probs[:, AA_TO_IDX[aa] + 1] for aa in AA_LIST], dim=-1)
            all_probs.append(aa_probs)
            next_token = probs.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return torch.stack(all_probs, dim=1)  # (B, length, NUM_AA)


# â”€â”€â”€ Datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EncDataset(Dataset):
    def __init__(self, ctx, cont, aug=True, factor=5):
        self.pairs = list(zip(ctx, cont))
        self.aug = aug
        self.factor = factor

    def __len__(self): return len(self.pairs) * (self.factor if self.aug else 1)

    def __getitem__(self, i):
        i = i % len(self.pairs)
        ctx, cont = self.pairs[i]
        if self.aug and random.random() < 0.7:
            full = ctx + cont
            lo, hi = max(20, len(full)//4), len(full) - CONT_LEN
            if hi > lo:
                sp = random.randint(lo, hi)
                ctx, cont = full[:sp], full[sp:sp+CONT_LEN]
        t = tok(ctx)
        if len(t) > MAX_LEN: t = t[-MAX_LEN:]
        return torch.tensor(t, dtype=torch.long), torch.tensor([AA_TO_IDX[c] for c in cont], dtype=torch.long)


class CLMDataset(Dataset):
    def __init__(self, sequences, aug=True, factor=3):
        self.seqs = sequences
        self.aug = aug
        self.factor = factor

    def __len__(self): return len(self.seqs) * (self.factor if self.aug else 1)

    def __getitem__(self, i):
        i = i % len(self.seqs)
        seq = self.seqs[i]
        if self.aug and random.random() < 0.5 and len(seq) > 60:
            start = random.randint(0, len(seq) // 4)
            end = random.randint(len(seq) * 3 // 4, len(seq))
            seq = seq[start:end]
        t = tok(seq)
        if len(t) > MAX_LEN + CONT_LEN:
            t = t[-(MAX_LEN + CONT_LEN):]
        inp = t[:-1]
        tgt = t[1:]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def enc_collate(batch):
    ss, ll = zip(*batch)
    ml = min(max(len(s) for s in ss), MAX_LEN)
    p = torch.zeros(len(ss), ml, dtype=torch.long)
    m = torch.ones(len(ss), ml, dtype=torch.bool)
    for i, s in enumerate(ss):
        L = min(len(s), ml)
        p[i, ml-L:] = s[-L:]
        m[i, ml-L:] = False
    return p, torch.stack(ll), m


def clm_collate(batch):
    inps, tgts = zip(*batch)
    ml = min(max(len(s) for s in inps), MAX_LEN + CONT_LEN)
    pi = torch.zeros(len(inps), ml, dtype=torch.long)
    pt = torch.zeros(len(tgts), ml, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inps, tgts)):
        L = min(len(inp), ml)
        pi[i, ml-L:] = inp[-L:]
        pt[i, ml-L:] = tgt[-L:]
    return pi, pt


# â”€â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def cosine_lr(step, warmup, peak, total):
    if step < warmup: return peak * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * t))


def train_encoder(tr_ctx, tr_cont, va_ctx, va_cont, seed=42, epochs=80, lr=3e-4,
                  d=192, heads=6, layers=4, ff=768, drop=0.25, batch_size=64):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model = Encoder(d, heads, layers, ff, drop).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    ds = EncDataset(tr_ctx, tr_cont, aug=True, factor=5)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=enc_collate, drop_last=True)
    total_steps = epochs * len(dl)
    warmup = int(0.05 * total_steps)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    step = best_acc = 0
    best_st = None; pat = 0

    for ep in range(epochs):
        model.train()
        for x, y, m in dl:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            clr = cosine_lr(step, warmup, lr, total_steps)
            for pg in opt.param_groups: pg['lr'] = clr
            loss = crit(model(x, m).reshape(-1, NUM_AA), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1

        if va_ctx and (ep+1) % 5 == 0:
            acc = eval_encoder(model, va_ctx, va_cont)
            if acc > best_acc:
                best_acc = acc; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
            else: pat += 1
            if (ep+1) % 10 == 0:
                print(f"    Ep {ep+1} | acc={acc:.4f} | best={best_acc:.4f}")
                sys.stdout.flush()
            if pat >= 8: print(f"    Stop at ep {ep+1}"); break

    if best_st: model.load_state_dict(best_st)
    print(f"    Final: {best_acc:.4f}"); sys.stdout.flush()
    return model, best_acc


def train_clm(tr_seqs, va_pairs, seed=42, epochs=60, lr=3e-4,
              d=192, heads=6, layers=4, ff=768, drop=0.25):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model = CausalLM(d, heads, layers, ff, drop).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
    ds = CLMDataset(tr_seqs, aug=True, factor=3)
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=clm_collate, drop_last=True)
    total_steps = epochs * len(dl)
    warmup = int(0.05 * total_steps)
    crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    step = best_acc = 0
    best_st = None; pat = 0

    for ep in range(epochs):
        model.train()
        for inp, tgt in dl:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            clr = cosine_lr(step, warmup, lr, total_steps)
            for pg in opt.param_groups: pg['lr'] = clr
            logits = model(inp)
            loss = crit(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1

        if va_pairs and (ep+1) % 5 == 0:
            acc = eval_clm(model, va_pairs)
            if acc > best_acc:
                best_acc = acc; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
            else: pat += 1
            if (ep+1) % 10 == 0:
                print(f"    CLM Ep {ep+1} | acc={acc:.4f} | best={best_acc:.4f}")
                sys.stdout.flush()
            if pat >= 8: print(f"    CLM Stop at ep {ep+1}"); break

    if best_st: model.load_state_dict(best_st)
    print(f"    CLM Final: {best_acc:.4f}"); sys.stdout.flush()
    return model, best_acc


def eval_encoder(model, ctx, cont):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for cx, co in zip(ctx, cont):
            ids = torch.tensor([tok(cx)], dtype=torch.long).to(DEVICE)
            if ids.size(1) > MAX_LEN: ids = ids[:, -MAX_LEN:]
            pr = model(ids)[0].argmax(-1)
            for i in range(CONT_LEN):
                if AA_LIST[pr[i].item()] == co[i]: c += 1
                t += 1
    return c / t


def eval_clm(model, va_pairs):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for cx, co in va_pairs:
            ids = torch.tensor([tok(cx)], dtype=torch.long).to(DEVICE)
            probs = model.get_continuation_probs(ids)
            preds = probs[0].argmax(-1)
            for i in range(CONT_LEN):
                if AA_LIST[preds[i].item()] == co[i]: c += 1
                t += 1
    return c / t


def get_enc_probs(model, ctx):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tok(ctx)], dtype=torch.long).to(DEVICE)
        if ids.size(1) > MAX_LEN: ids = ids[:, -MAX_LEN:]
        return F.softmax(model(ids)[0], dim=-1).cpu().numpy()


def get_clm_probs(model, ctx):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tok(ctx)], dtype=torch.long).to(DEVICE)
        probs = model.get_continuation_probs(ids)
        return probs[0].cpu().numpy()


# â”€â”€â”€ Ensemble â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def tune_ensemble_poswise(model_probs_list, va_cont):
    """Tune per-position weights for ensemble."""
    n_models = len(model_probs_list)
    N = len(va_cont)

    # First find global best weights
    best_acc = best_weights = 0
    step = 1.0 / 20  # finer grid
    for w_step in np.arange(0, 1.01, 0.05):
        for i in range(n_models):
            # Try giving extra weight to model i
            weights = np.ones(n_models) * (1.0 - w_step) / max(1, n_models - 1)
            weights[i] = w_step
            if n_models == 1:
                weights = np.array([1.0])

            c = t = 0
            for j in range(N):
                combined = sum(weights[m] * model_probs_list[m][j] for m in range(n_models))
                for pos in range(CONT_LEN):
                    if AA_LIST[np.argmax(combined[pos])] == va_cont[j][pos]: c += 1
                    t += 1
            acc = c / t
            if acc > best_acc:
                best_acc = acc
                best_weights = weights.copy()

    # Also try equal weights
    weights = np.ones(n_models) / n_models
    c = t = 0
    for j in range(N):
        combined = sum(weights[m] * model_probs_list[m][j] for m in range(n_models))
        for pos in range(CONT_LEN):
            if AA_LIST[np.argmax(combined[pos])] == va_cont[j][pos]: c += 1
            t += 1
    acc = c / t
    if acc > best_acc:
        best_acc = acc
        best_weights = weights.copy()

    print(f"  Global best: weights={[f'{w:.3f}' for w in best_weights]} -> {best_acc:.4f}")
    return best_weights, best_acc


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    t0 = time.time()
    print(f"Device: {DEVICE}"); sys.stdout.flush()

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    contexts = train_df['context'].tolist()
    conts = train_df['continuation'].tolist()
    full_seqs = [c + t for c, t in zip(contexts, conts)]
    test_ctx = test_df['context'].tolist()
    all_seqs = full_seqs + test_ctx

    # Split
    np.random.seed(SEED)
    idx = np.random.permutation(len(contexts))
    vn = int(0.15 * len(contexts))
    vi, ti = idx[:vn], idx[vn:]
    tr_ctx = [contexts[i] for i in ti]
    tr_cont = [conts[i] for i in ti]
    tr_full = [full_seqs[i] for i in ti]
    va_ctx = [contexts[i] for i in vi]
    va_cont = [conts[i] for i in vi]
    va_pairs = list(zip(va_ctx, va_cont))

    # N-gram
    print("=== N-gram ==="); sys.stdout.flush()
    ngram = NgramModel()
    ngram.fit(all_seqs)

    # Train diverse models
    print("\n=== Training Models ==="); sys.stdout.flush()

    enc_configs = [
        dict(seed=42,  d=192, heads=6, layers=4, ff=768,  drop=0.25, lr=3e-4, epochs=80),
        dict(seed=123, d=128, heads=4, layers=6, ff=512,  drop=0.30, lr=2e-4, epochs=80),
        dict(seed=456, d=256, heads=8, layers=3, ff=1024, drop=0.20, lr=4e-4, epochs=80),
        dict(seed=789, d=160, heads=8, layers=5, ff=640,  drop=0.28, lr=3e-4, epochs=80),
        dict(seed=101, d=224, heads=8, layers=4, ff=896,  drop=0.22, lr=3e-4, epochs=80),
    ]

    clm_configs = [
        dict(seed=42,  d=192, heads=6, layers=4, ff=768,  drop=0.25, lr=3e-4, epochs=60),
        dict(seed=321, d=160, heads=8, layers=5, ff=640,  drop=0.28, lr=2e-4, epochs=60),
    ]

    all_va_probs = []
    all_models_info = []

    # Train encoders
    for ci, cfg in enumerate(enc_configs):
        print(f"\n  Encoder {ci}: d={cfg['d']} L={cfg['layers']} drop={cfg['drop']}")
        sys.stdout.flush()
        m, acc = train_encoder(tr_ctx, tr_cont, va_ctx, va_cont, **cfg)
        vp = np.array([get_enc_probs(m, cx) for cx in va_ctx])
        all_va_probs.append(vp)
        all_models_info.append(('enc', cfg, acc))
        del m; torch.cuda.empty_cache()

    # Train CLMs
    for ci, cfg in enumerate(clm_configs):
        print(f"\n  CLM {ci}: d={cfg['d']} L={cfg['layers']} drop={cfg['drop']}")
        sys.stdout.flush()
        m, acc = train_clm(tr_full, va_pairs, **cfg)
        vp = np.array([get_clm_probs(m, cx) for cx, _ in va_pairs])
        all_va_probs.append(vp)
        all_models_info.append(('clm', cfg, acc))
        del m; torch.cuda.empty_cache()

    # N-gram probs
    ng_va = np.array([ngram.continuation_probs(cx) for cx in va_ctx])
    all_va_probs.append(ng_va)
    all_models_info.append(('ngram', {}, 0))

    # Print individual model accuracies
    print("\n=== Individual Accuracies ===")
    n_total = len(all_va_probs)
    for mi in range(n_total):
        c = sum(1 for j in range(len(va_ctx)) for i in range(CONT_LEN)
                if AA_LIST[np.argmax(all_va_probs[mi][j][i])] == va_cont[j][i])
        mtype = all_models_info[mi][0]
        print(f"  {mtype} {mi}: {c/(len(va_ctx)*CONT_LEN):.4f}")
    sys.stdout.flush()

    # Ensemble tuning
    print("\n=== Ensemble Tuning ==="); sys.stdout.flush()
    best_weights, best_acc = tune_ensemble_poswise(all_va_probs, va_cont)

    # Also try removing each model
    n_all = len(all_va_probs)
    for drop_m in range(n_all):
        remaining = [m for m in range(n_all) if m != drop_m]
        sub_probs = [all_va_probs[m] for m in remaining]
        w, a = tune_ensemble_poswise(sub_probs, va_cont)
        if a > best_acc:
            best_acc = a
            # Reconstruct full weight vector with 0 for dropped model
            full_w = np.zeros(n_all)
            for ri, mi in enumerate(remaining):
                full_w[mi] = w[ri]
            best_weights = full_w
            print(f"  Better w/o model {drop_m} ({all_models_info[drop_m][0]}): {a:.4f}")

    print(f"\nBest val: {best_acc:.4f}")
    print(f"Weights: {[f'{w:.3f}' for w in best_weights]}")
    sys.stdout.flush()

    # â”€â”€ Retrain on ALL data â”€â”€
    print("\n=== Retrain Full ==="); sys.stdout.flush()
    ngram_full = NgramModel()
    ngram_full.fit(all_seqs)

    full_models = []
    for mi, (mtype, cfg, _) in enumerate(all_models_info):
        if best_weights[mi] < 0.01:
            full_models.append(None)
            print(f"  Skipping model {mi} (weight={best_weights[mi]:.3f})")
            continue
        if mtype == 'enc':
            print(f"\n  Full encoder {mi}")
            sys.stdout.flush()
            m, _ = train_encoder(contexts, conts, None, None, **{**cfg, 'epochs': 50})
            full_models.append(('enc', m))
        elif mtype == 'clm':
            print(f"\n  Full CLM {mi}")
            sys.stdout.flush()
            m, _ = train_clm(full_seqs, None, **{**cfg, 'epochs': 40})
            full_models.append(('clm', m))
        else:
            full_models.append(('ngram', None))

    # â”€â”€ Predict â”€â”€
    print("\n=== Predict ==="); sys.stdout.flush()
    preds = []
    for j, cx in enumerate(test_ctx):
        combined = np.zeros((CONT_LEN, NUM_AA))
        for mi in range(n_all):
            w = best_weights[mi]
            if w < 0.01: continue
            if full_models[mi] is None: continue
            mtype, m = full_models[mi]
            if mtype == 'enc':
                p = get_enc_probs(m, cx)
            elif mtype == 'clm':
                p = get_clm_probs(m, cx)
            else:
                p = ngram_full.continuation_probs(cx)
            combined += w * p

        preds.append(''.join(AA_LIST[np.argmax(combined[i])] for i in range(CONT_LEN)))
        if (j+1) % 100 == 0: print(f"  {j+1}/{len(test_ctx)}"); sys.stdout.flush()

    sub = pd.DataFrame({'seq_id': test_df['seq_id'], 'continuation': preds})
    out = os.path.join(WORKING_DIR, 'submission.csv')
    sub.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(f"Val: {best_acc:.4f} | Time: {(time.time()-t0)/60:.1f}min")
    print(sub.head())
    sys.stdout.flush()


if __name__ == '__main__':
    main()
