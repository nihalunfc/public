from __future__ import annotations

import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SEED = 42
SOLUTION_DIR = Path(__file__).resolve().parent
WORKING_DIR = SOLUTION_DIR / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
TEXT_PROBS_PATH = WORKING_DIR / "text_probs.npz"


def locate_data_dir() -> Path:
    for candidate in [
        SOLUTION_DIR / "dataset" / "public",
        SOLUTION_DIR / "dataset",
        SOLUTION_DIR / "public",
        SOLUTION_DIR / "data",
    ]:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate dataset directory")


DATA_DIR = locate_data_dir()

TRANSIENT_CLASSES = [
    "Cryogenic Afterglow", "Dark Reverberator", "Helicity Collapse",
    "Hot Jet Eruption", "Hyperaccretion Flare", "Limb-Cycle Pulsar",
    "Lithogen Burst", "Neutronfall", "Quasi-Periodic Echoer",
    "Spectral Ghost", "Tidal Spectacle", "Tidally Locked Beacon",
]
HOST_ENVIRONMENTS = [
    "AGN Wind Region", "Circumnuclear Disk", "Diffuse Warm Medium",
    "Galactic Bar Vicinity", "Globular Cluster Core", "Intergalactic Halo",
    "Nuclear Star Cluster", "Stellar Stream", "Young Stellar Association",
]
SPECTRAL_PATTERNS = {
    "hard_xray": [r"hard[\s_-]?x[\s_-]?ray"],
    "soft_xray": [r"soft[\s_-]?x[\s_-]?ray"],
    "optical": [r"\boptical\b"],
    "radio": [r"\bradio\b"],
    "infrared": [r"\binfrared\b", r"\bir\s+spectral"],
    "uv": [r"\buv\b", r"ultraviolet"],
}
VARIABILITY_PHRASES = {
    "chaotic": [r"chaotic"],
    "double_peak": [r"double[\s-]?peak", r"two[\s-]?peak", r"double peaks"],
    "flat": [
        r"\bflat\b", r"steady (state|emission|brightness)",
        r"constant (brightness|emission|luminosity|flux|intensity)",
        r"no (significant )?(variab|fluct|change)",
    ],
    "monotonic_rise": [
        r"monotonic", r"steady rise", r"continuous(ly)? (ris|increas)",
        r"steadily (increas|ris)", r"gradual(ly)? (ris|increas)",
    ],
    "quasi_periodic": [r"quasi[\s-]?period"],
}

CLEAN_LABELS = ["transient_class", "host_environment", "spectral_regime"]
SCORED_LABELS = [
    "variability_pattern", "distance_bin", "energy_tier",
    "followup_protocol", "precursor_category",
]
ALL_PRED_LABELS = CLEAN_LABELS + SCORED_LABELS
SUBMISSION_COLUMNS = ["id"] + ALL_PRED_LABELS

CAT_FEATURES = ["x_class", "x_env", "x_spec", "x_vp"]
NUM_FEATURES = ["x_z", "x_L", "x_z_miss", "x_L_miss", "x_z_log", "x_L_log", "x_nw"]

CB_SEEDS = (42, 2024, 7)
CB_ITERS_SCORED = 500
CB_ITERS_CLEAN = 300
N_OOF_FOLDS = 3

TX_MODEL_NAME = os.environ.get("TEXT_MODEL", "distilbert-base-uncased")
TX_MAX_LEN = int(os.environ.get("TEXT_MAX_LEN", "192"))
TX_BATCH = int(os.environ.get("TEXT_BATCH", "32"))
TX_BATCH_EVAL = int(os.environ.get("TEXT_BATCH_EVAL", "64"))
TX_EPOCHS = int(os.environ.get("TEXT_EPOCHS", "4"))
TX_LR = float(os.environ.get("TEXT_LR", "3e-5"))
TX_WD = float(os.environ.get("TEXT_WD", "0.01"))
TX_WARMUP = float(os.environ.get("TEXT_WARMUP", "0.06"))
TX_SEED = int(os.environ.get("TEXT_SEED", "42"))
TX_AMP = os.environ.get("TEXT_AMP", "1") == "1"


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_clean_labels(narr: str) -> tuple[str, str, str]:
    nlow = narr.lower()
    cls = next((c for c in TRANSIENT_CLASSES if c.lower() in nlow), "")
    env = next((e for e in HOST_ENVIRONMENTS if e.lower() in nlow), "")
    spec = ""
    for name, pats in SPECTRAL_PATTERNS.items():
        if any(re.search(p, nlow) for p in pats):
            spec = name
            break
    return cls, env, spec


def extract_numeric(narr: str) -> tuple[float, float]:
    z_match = re.search(r"z\s*=\s*(-?\d+\.?\d*)", narr)
    l_match = re.search(r"log\s*L\s*=\s*(-?\d+\.?\d*)", narr)
    z = float(z_match.group(1)) if z_match else -1.0
    L = float(l_match.group(1)) if l_match else -1.0
    return z, L


def detect_variability(narr: str) -> str:
    nlow = narr.lower()
    hits = []
    for vp, pats in VARIABILITY_PHRASES.items():
        if any(re.search(p, nlow) for p in pats):
            hits.append(vp)
    return hits[0] if len(hits) == 1 else "unknown"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    clean = df["narrative"].apply(extract_clean_labels)
    out["x_class"] = clean.apply(lambda t: t[0] or "UNK")
    out["x_env"] = clean.apply(lambda t: t[1] or "UNK")
    out["x_spec"] = clean.apply(lambda t: t[2] or "UNK")
    nums = df["narrative"].apply(extract_numeric)
    out["x_z"] = nums.apply(lambda t: t[0]).astype(float)
    out["x_L"] = nums.apply(lambda t: t[1]).astype(float)
    out["x_z_miss"] = (out["x_z"] == -1.0).astype(float)
    out["x_L_miss"] = (out["x_L"] == -1.0).astype(float)
    with np.errstate(divide="ignore"):
        z_safe = out["x_z"].values
        out["x_z_log"] = np.where(z_safe > 0, np.log1p(z_safe), -1.0)
        l_safe = out["x_L"].values
        out["x_L_log"] = np.where(l_safe > 0, np.log1p(l_safe), -1.0)
    out["x_vp"] = df["narrative"].apply(detect_variability)
    out["x_nw"] = df["narrative"].str.split().str.len().astype(float)
    return out


def build_text_features(train_narr, test_narr):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.95,
        sublinear_tf=True, max_features=15000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.95,
        sublinear_tf=True, max_features=8000,
    )
    w_tr = word_vec.fit_transform(train_narr)
    w_te = word_vec.transform(test_narr)
    c_tr = char_vec.fit_transform(train_narr)
    c_te = char_vec.transform(test_narr)
    return hstack([w_tr, c_tr]).tocsr(), hstack([w_te, c_te]).tocsr()


def adversarial_weights(X_tr: np.ndarray, X_te: np.ndarray,
                        seed: int = 0) -> np.ndarray:
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    lr = LogisticRegression(max_iter=400, solver="lbfgs", C=1.0, random_state=seed)
    lr.fit(X, y)
    p = lr.predict_proba(X_tr)[:, 1]
    w = p / (1 - p + 1e-6)
    cap = np.quantile(w, 0.98)
    w = np.clip(w, 0.0, cap)
    w = w / (w.mean() + 1e-9)
    return w


def train_cb(X_tr: pd.DataFrame, y_tr: np.ndarray, iters: int, seed: int,
             cat_features: list[str],
             sample_weight: np.ndarray | None = None) -> CatBoostClassifier:
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_features if c in X_tr.columns]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iters,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr, cat_features=cat_idx, sample_weight=sample_weight)
    return model


def align_proba(model, X, classes: list) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((raw.shape[0], len(classes)))
    for i, c in enumerate(model.classes_):
        out[:, classes.index(c if not isinstance(c, np.generic) else c.item())] = raw[:, i]
    return out


def simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    out: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


def optimize_blend(y_true: np.ndarray, classes: list,
                   p_cb: np.ndarray, p_lr: np.ndarray,
                   p_tx: np.ndarray | None = None,
                   ) -> tuple[tuple[float, float, float], float]:
    ca = np.array(classes)
    if p_tx is None:
        best_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
        best_f1 = -1.0
        for w_cb in np.arange(0.0, 1.001, 0.05):
            w_lr = round(1.0 - w_cb, 4)
            pred = ca[(w_cb * p_cb + w_lr * p_lr).argmax(1)]
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_w = f1, (w_cb, w_lr, 0.0)
        return best_w, best_f1
    best_w = (1.0, 0.0, 0.0)
    best_f1 = -1.0
    for w in simplex_grid(0.05):
        pred = ca[(w[0] * p_cb + w[1] * p_lr + w[2] * p_tx).argmax(1)]
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def _torch_seed(s: int) -> None:
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _build_tx_model(label_classes):
    import torch
    from torch import nn
    from transformers import AutoModel

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(TX_MODEL_NAME)
            h = self.encoder.config.hidden_size
            self.drop = nn.Dropout(0.1)
            self.heads = nn.ModuleList([nn.Linear(h, k) for k in label_classes])

        def forward(self, input_ids, attention_mask):
            o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            x = self.drop(o.last_hidden_state[:, 0, :])
            return [head(x) for head in self.heads]

    return _M()


def _tx_fit(model, ids_tr, mask_tr, y_tr, device, epochs, lr, wd,
            warmup, bs, use_amp):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup as glsw

    model.to(device)
    ds = TensorDataset(ids_tr, mask_tr, y_tr)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
    total = max(1, len(loader) * epochs)
    nd = {"bias", "LayerNorm.weight"}
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(d in n for d in nd)], "weight_decay": wd},
        {"params": [p for n, p in model.named_parameters()
                    if any(d in n for d in nd)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = glsw(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        rl = 0.0
        for batch in loader:
            b_ids, b_mask, b_y = [t.to(device, non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(b_ids, b_mask)
                loss = sum(loss_fn(logits[k], b_y[:, k]) for k in range(len(logits)))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            rl += loss.item()
        log(f"    ep {ep+1}/{epochs} loss={rl/len(loader):.4f}")


def _tx_predict(model, ids, mask, device, bs, use_amp):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    loader = DataLoader(TensorDataset(ids, mask), batch_size=bs,
                        shuffle=False, pin_memory=True)
    hp = None
    with torch.no_grad():
        for batch in loader:
            b_ids, b_mask = [t.to(device, non_blocking=True) for t in batch]
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(b_ids, b_mask)
            if hp is None:
                hp = [[] for _ in logits]
            for k, lo in enumerate(logits):
                hp[k].append(torch.softmax(lo.float(), dim=-1).cpu().numpy())
    return [np.concatenate(b, axis=0) for b in hp]


def compute_text_probs(train: pd.DataFrame, test: pd.DataFrame,
                       save: bool = True) -> dict | None:
    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as e:
        log(f"  TX: unavailable ({e}); skipping")
        return None
    if not torch.cuda.is_available():
        log("  TX: no CUDA; skipping")
        return None

    t0 = time.time()
    _torch_seed(TX_SEED)
    device = torch.device("cuda")
    log(f"  TX: model={TX_MODEL_NAME} len={TX_MAX_LEN} ep={TX_EPOCHS} "
        f"bs={TX_BATCH} amp={TX_AMP}")

    cpl: dict = {}
    lc: list = []
    for col in SCORED_LABELS:
        c = sorted(train[col].unique().tolist())
        cpl[col] = c
        lc.append(len(c))

    n_tr = len(train)
    y_int = np.zeros((n_tr, len(SCORED_LABELS)), dtype=np.int64)
    for j, col in enumerate(SCORED_LABELS):
        mp = {c: i for i, c in enumerate(cpl[col])}
        y_int[:, j] = train[col].map(mp).values

    tok = AutoTokenizer.from_pretrained(TX_MODEL_NAME)
    enc_tr = tok(train["narrative"].tolist(), padding="max_length",
                 truncation=True, max_length=TX_MAX_LEN, return_tensors="pt")
    enc_te = tok(test["narrative"].tolist(), padding="max_length",
                 truncation=True, max_length=TX_MAX_LEN, return_tensors="pt")
    ids_tr, msk_tr = enc_tr["input_ids"], enc_tr["attention_mask"]
    ids_te, msk_te = enc_te["input_ids"], enc_te["attention_mask"]
    y_tensor = torch.from_numpy(y_int)

    pair = (train["transient_class"] + "||" + train["host_environment"]).values
    oof = [np.zeros((n_tr, k), dtype=np.float32) for k in lc]
    te_folds: list = [[] for _ in SCORED_LABELS]

    gkf = GroupKFold(n_splits=N_OOF_FOLDS)
    for fold, (ti, vi) in enumerate(gkf.split(train, groups=pair)):
        ft0 = time.time()
        log(f"  TX fold {fold+1}/{N_OOF_FOLDS} train={len(ti)} val={len(vi)}")
        _torch_seed(TX_SEED + fold)
        mdl = _build_tx_model(lc)
        _tx_fit(mdl, ids_tr[ti], msk_tr[ti], y_tensor[ti], device,
                TX_EPOCHS, TX_LR, TX_WD, TX_WARMUP, TX_BATCH, TX_AMP)
        vp = _tx_predict(mdl, ids_tr[vi], msk_tr[vi], device, TX_BATCH_EVAL, TX_AMP)
        tp = _tx_predict(mdl, ids_te, msk_te, device, TX_BATCH_EVAL, TX_AMP)
        for k in range(len(SCORED_LABELS)):
            oof[k][vi] = vp[k]
            te_folds[k].append(tp[k])
        del mdl
        torch.cuda.empty_cache()
        log(f"    fold time={time.time()-ft0:.1f}s")

    te_mean = [np.mean(np.stack(fs, 0), 0) for fs in te_folds]

    out: dict = {}
    for k, col in enumerate(SCORED_LABELS):
        out[col] = {"oof": oof[k], "test": te_mean[k], "classes": cpl[col]}

    if save:
        try:
            sd: dict = {}
            for k, col in enumerate(SCORED_LABELS):
                sd[f"oof__{col}"] = oof[k]
                sd[f"test__{col}"] = te_mean[k]
                sd[f"classes__{col}"] = np.array(cpl[col], dtype=object)
            sd["train_ids"] = train["id"].values
            sd["test_ids"] = test["id"].values
            np.savez(TEXT_PROBS_PATH, **sd)
            log(f"  TX: saved {TEXT_PROBS_PATH}")
        except Exception as e:
            log(f"  TX: save failed: {e}")

    log(f"  TX total={time.time()-t0:.1f}s")
    return out


def load_text_probs(train: pd.DataFrame, test: pd.DataFrame) -> dict | None:
    if not TEXT_PROBS_PATH.exists():
        return None
    cache = np.load(TEXT_PROBS_PATH, allow_pickle=True)
    if not np.array_equal(cache["train_ids"], train["id"].values):
        return None
    if not np.array_equal(cache["test_ids"], test["id"].values):
        return None
    out: dict = {}
    for col in SCORED_LABELS:
        oof = cache[f"oof__{col}"]
        te = cache[f"test__{col}"]
        tx_cls = list(cache[f"classes__{col}"])
        sol_cls = sorted(train[col].unique().tolist())
        if tx_cls != sol_cls:
            return None
        out[col] = {"oof": oof.astype(np.float32),
                    "test": te.astype(np.float32),
                    "classes": sol_cls}
    log(f"  TX: loaded cache from {TEXT_PROBS_PATH}")
    return out


def main() -> None:
    seed_everything(SEED)
    t0 = time.time()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    log(f"train={train.shape} test={test.shape} sample={sample_sub.shape}")

    feats_tr = build_features(train)
    feats_te = build_features(test)

    for col, xcol in [("transient_class", "x_class"), ("host_environment", "x_env"),
                      ("spectral_regime", "x_spec")]:
        log(f"  extraction {col}: {(feats_tr[xcol] == train[col]).mean():.4f}")

    tf_tr, tf_te = build_text_features(train["narrative"], test["narrative"])
    X_tab_tr = feats_tr[CAT_FEATURES + NUM_FEATURES].reset_index(drop=True)
    X_tab_te = feats_te[CAT_FEATURES + NUM_FEATURES].reset_index(drop=True)
    log(f"  tab={X_tab_tr.shape} tfidf={tf_tr.shape} setup={time.time()-t0:.1f}s")

    text_probs = None
    if os.environ.get("SOLUTION_USE_TEXT", "1") == "1":
        text_probs = load_text_probs(train, test)
        if text_probs is None:
            log("  TX cache miss; training transformer")
            text_probs = compute_text_probs(train, test, save=True)
            if text_probs is None:
                log("  continuing without transformer")

    pair = (train["transient_class"] + "||" + train["host_environment"]).values
    gkf = GroupKFold(n_splits=N_OOF_FOLDS)

    log("OOF phase: collecting CB+LR predictions for blend optimization")
    oof_cb: dict = {}
    oof_lr: dict = {}
    for col in SCORED_LABELS:
        classes = sorted(train[col].unique().tolist())
        y = train[col].values
        nc = len(classes)
        p_cb_oof = np.zeros((len(train), nc), dtype=np.float32)
        p_lr_oof = np.zeros((len(train), nc), dtype=np.float32)

        for fold, (ti, vi) in enumerate(gkf.split(train, groups=pair)):
            X_tr_f = X_tab_tr.iloc[ti].reset_index(drop=True)
            X_va_f = X_tab_tr.iloc[vi].reset_index(drop=True)
            tr_num = X_tr_f[NUM_FEATURES].values.astype(np.float32)
            va_num = X_va_f[NUM_FEATURES].values.astype(np.float32)
            w_fold = adversarial_weights(tr_num, va_num, seed=fold)

            for s in CB_SEEDS:
                m = train_cb(X_tr_f, y[ti], iters=CB_ITERS_SCORED, seed=s,
                             cat_features=CAT_FEATURES, sample_weight=w_fold)
                p_cb_oof[vi] += align_proba(m, X_va_f, classes)
            p_cb_oof[vi] /= len(CB_SEEDS)

            lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                    n_jobs=-1, random_state=SEED)
            lr.fit(tf_tr[ti], y[ti], sample_weight=w_fold)
            p_lr_oof[vi] = align_proba(lr, tf_tr[vi], classes)

        oof_cb[col] = p_cb_oof
        oof_lr[col] = p_lr_oof
        f_cb = f1_score(y, np.array(classes)[p_cb_oof.argmax(1)],
                        average="macro", zero_division=0)
        f_lr = f1_score(y, np.array(classes)[p_lr_oof.argmax(1)],
                        average="macro", zero_division=0)
        log(f"  {col} OOF: cb={f_cb:.4f} lr={f_lr:.4f}")

    log("simplex blend optimization per label")
    blend_weights: dict = {}
    for col in SCORED_LABELS:
        classes = sorted(train[col].unique().tolist())
        y = train[col].values
        p_tx = text_probs[col]["oof"] if text_probs and col in text_probs else None
        w, f1 = optimize_blend(y, classes, oof_cb[col], oof_lr[col], p_tx)
        blend_weights[col] = w
        log(f"  {col}: w=(cb={w[0]:.2f} lr={w[1]:.2f} tx={w[2]:.2f}) F1={f1:.4f}")

    oof_mean = np.mean([
        f1_score(train[col].values,
                 np.array(sorted(train[col].unique().tolist()))[
                     (blend_weights[col][0] * oof_cb[col]
                      + blend_weights[col][1] * oof_lr[col]
                      + (blend_weights[col][2] * text_probs[col]["oof"]
                         if text_probs and col in text_probs and blend_weights[col][2] > 0
                         else 0)
                      ).argmax(1)],
                 average="macro", zero_division=0)
        for col in SCORED_LABELS
    ])
    log(f"  OOF mean F1 (optimized blend) = {oof_mean:.4f}")

    log("computing adversarial weights full train vs test")
    tr_num = X_tab_tr[NUM_FEATURES].values.astype(np.float32)
    te_num = X_tab_te[NUM_FEATURES].values.astype(np.float32)
    weights = adversarial_weights(tr_num, te_num, seed=0)
    log(f"  w stats: min={weights.min():.3f} mean={weights.mean():.3f} "
        f"max={weights.max():.3f}")

    log("fitting final models on full train")
    submission = pd.DataFrame({"id": test["id"].values})

    for col in CLEAN_LABELS:
        t1 = time.time()
        classes = sorted(train[col].unique().tolist())
        m = train_cb(X_tab_tr, train[col].values, iters=CB_ITERS_CLEAN,
                     seed=42, cat_features=CAT_FEATURES)
        submission[col] = np.array(classes)[align_proba(m, X_tab_te, classes).argmax(1)]
        log(f"  {col}: {time.time()-t1:.1f}s")

    for col in SCORED_LABELS:
        t1 = time.time()
        classes = sorted(train[col].unique().tolist())
        y_tr = train[col].values

        p_cb_list = []
        for s in CB_SEEDS:
            m = train_cb(X_tab_tr, y_tr, iters=CB_ITERS_SCORED, seed=s,
                         cat_features=CAT_FEATURES, sample_weight=weights)
            p_cb_list.append(align_proba(m, X_tab_te, classes))
        p_cb = np.mean(p_cb_list, axis=0)

        lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                n_jobs=-1, random_state=SEED)
        lr.fit(tf_tr, y_tr, sample_weight=weights)
        p_lr = align_proba(lr, tf_te, classes)

        p_tx = None
        if text_probs is not None and col in text_probs:
            p_tx = text_probs[col]["test"]

        w = blend_weights[col]
        blend = w[0] * p_cb + w[1] * p_lr
        if p_tx is not None and w[2] > 0:
            blend = blend + w[2] * p_tx

        submission[col] = np.array(classes)[blend.argmax(1)]
        log(f"  {col}: {time.time()-t1:.1f}s  w=(cb={w[0]:.2f} lr={w[1]:.2f} tx={w[2]:.2f})")

    submission = submission[SUBMISSION_COLUMNS]
    assert list(submission.columns) == list(sample_sub.columns), "column mismatch"
    assert len(submission) == len(sample_sub), "row count mismatch"
    assert set(submission["id"]) == set(sample_sub["id"]), "id set mismatch"
    submission.to_csv(SUBMISSION_PATH, index=False)
    assert SUBMISSION_PATH.exists() and SUBMISSION_PATH.stat().st_size > 0
    log(f"wrote {SUBMISSION_PATH} shape={submission.shape}")
    log(submission.head().to_string(index=False))
    log(f"TOTAL elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
