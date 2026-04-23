from __future__ import annotations

import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - fallback defensivo
    Ridge = None
    Pipeline = None
    StandardScaler = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - fallback defensivo
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "public"
WORKING_DIR = ROOT / "working"
CACHE_DIR = WORKING_DIR / "cache"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
REPORT_PATH = WORKING_DIR / "training_report.json"

RANDOM_STATE = 42
PREDICTION_LENGTH = 20
ALPHABET = "ACDEFGHIKLMNPQRSTVWYU"
ALPHABET_SET = set(ALPHABET)
CHAR_TO_INDEX = {
    character: index + 1 for index, character in enumerate(ALPHABET)
}
INDEX_TO_CHAR = {
    index + 1: character for index, character in enumerate(ALPHABET)
}
TARGET_TO_CHAR = {index: character for index, character in enumerate(ALPHABET)}
PAD_INDEX = 0
DEFAULT_CONTEXT_TOKEN = CHAR_TO_INDEX["A"]

MARKOV_FALLBACK_ORDER = 2

TRANSFORMER_QUERY_FAMILY_ORDER = ("v10", "v11b", "v8")
TRANSFORMER_QUERY_PRIMARY_FAMILY = "v10"
TRANSFORMER_QUERY_FAMILY_WEIGHTS = {
    "v7": 0.0,
    "v8": 1.0,
    "v9": 0.0,
    "v10": 0.8,
    "v11b": 0.2,
}
TRANSFORMER_QUERY_FAMILY_SPLIT_SEEDS = {
    "v7": (11, 42, 73, 7, 101),
    "v8": (11, 42, 73, 7, 101),
    "v9": (11, 42, 73),
    "v10": (11, 42, 73),
    "v11b": (11, 42, 73),
}
TRANSFORMER_QUERY_FAMILY_CONFIGS = {
    "v7": {
        "family_name": "v7",
        "cache_prefix_root": "transformer_query_ensemble_v7",
        "max_len": 192,
        "d_model": 96,
        "nhead": 4,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.2,
        "epochs": 12,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosineannealinglr",
        "use_padding_masks": False,
    },
    "v8": {
        "family_name": "v8",
        "cache_prefix_root": "transformer_query_ensemble_v8",
        "max_len": 256,
        "d_model": 128,
        "nhead": 8,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.1,
        "epochs": 16,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosineannealinglr",
        "use_padding_masks": False,
    },
    "v9": {
        "family_name": "v9",
        "cache_prefix_root": "transformer_query_ensemble_v9",
        "max_len": 320,
        "d_model": 128,
        "nhead": 8,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.1,
        "epochs": 12,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosineannealinglr",
        "use_padding_masks": False,
    },
    "v10": {
        "family_name": "v10",
        "cache_prefix_root": "transformer_query_ensemble_v10",
        "max_len": 256,
        "d_model": 128,
        "nhead": 8,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.1,
        "epochs": 4,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosineannealinglr",
        "use_padding_masks": False,
        "synthetic_prefix_stride": 16,
        "synthetic_min_prefix": 48,
        "synthetic_pretrain_epochs": 2,
        "synthetic_finetune_epochs": 2,
    },
    "v11b": {
        "family_name": "v11b",
        "cache_prefix_root": "transformer_query_probe_v11b",
        "max_len": 256,
        "d_model": 128,
        "nhead": 8,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dropout": 0.1,
        "epochs": 4,
        "lr": 1.5e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "scheduler": "cosineannealinglr",
        "use_padding_masks": False,
        "synthetic_prefix_stride": 16,
        "synthetic_min_prefix": 160,
        "synthetic_pretrain_epochs": 2,
        "synthetic_finetune_epochs": 2,
    },
}
DEFAULT_TRANSFORMER_QUERY_CONFIG = dict(
    TRANSFORMER_QUERY_FAMILY_CONFIGS[TRANSFORMER_QUERY_PRIMARY_FAMILY]
)

TRANSFORMER_QUERY_MAX_LEN = DEFAULT_TRANSFORMER_QUERY_CONFIG["max_len"]
TRANSFORMER_QUERY_D_MODEL = DEFAULT_TRANSFORMER_QUERY_CONFIG["d_model"]
TRANSFORMER_QUERY_NHEAD = DEFAULT_TRANSFORMER_QUERY_CONFIG["nhead"]
TRANSFORMER_QUERY_ENCODER_LAYERS = DEFAULT_TRANSFORMER_QUERY_CONFIG[
    "encoder_layers"
]
TRANSFORMER_QUERY_DECODER_LAYERS = DEFAULT_TRANSFORMER_QUERY_CONFIG[
    "decoder_layers"
]
TRANSFORMER_QUERY_DROPOUT = DEFAULT_TRANSFORMER_QUERY_CONFIG["dropout"]
TRANSFORMER_QUERY_EPOCHS = DEFAULT_TRANSFORMER_QUERY_CONFIG["epochs"]
TRANSFORMER_QUERY_BATCH_SIZE_CPU = 16
TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED = 32
TRANSFORMER_QUERY_LR = DEFAULT_TRANSFORMER_QUERY_CONFIG["lr"]
TRANSFORMER_QUERY_WEIGHT_DECAY = DEFAULT_TRANSFORMER_QUERY_CONFIG[
    "weight_decay"
]
TRANSFORMER_QUERY_GRAD_CLIP = DEFAULT_TRANSFORMER_QUERY_CONFIG["grad_clip"]
TRANSFORMER_QUERY_SCHEDULER = DEFAULT_TRANSFORMER_QUERY_CONFIG["scheduler"]
TRANSFORMER_QUERY_USE_PADDING_MASKS = DEFAULT_TRANSFORMER_QUERY_CONFIG[
    "use_padding_masks"
]
TRANSFORMER_QUERY_ENSEMBLE_SPLIT_SEEDS = TRANSFORMER_QUERY_FAMILY_SPLIT_SEEDS[
    TRANSFORMER_QUERY_PRIMARY_FAMILY
]

MODEL_CACHE_PREFIX = (
    f"{DEFAULT_TRANSFORMER_QUERY_CONFIG['cache_prefix_root']}"
    f"_seed{RANDOM_STATE}"
    f"_maxlen{TRANSFORMER_QUERY_MAX_LEN}"
    f"_dmodel{TRANSFORMER_QUERY_D_MODEL}"
    f"_nhead{TRANSFORMER_QUERY_NHEAD}"
    f"_enc{TRANSFORMER_QUERY_ENCODER_LAYERS}"
    f"_dec{TRANSFORMER_QUERY_DECODER_LAYERS}"
    f"_dropout{str(TRANSFORMER_QUERY_DROPOUT).replace('.', 'p')}"
    f"_epochs{TRANSFORMER_QUERY_EPOCHS}"
    f"_lr{str(TRANSFORMER_QUERY_LR).replace('.', 'p')}"
    f"_wd{str(TRANSFORMER_QUERY_WEIGHT_DECAY).replace('.', 'p')}"
    f"_scheduler{TRANSFORMER_QUERY_SCHEDULER}"
    f"_mask{int(TRANSFORMER_QUERY_USE_PADDING_MASKS)}"
)

CANDIDATE_BANK_VERSION = "v1"
CANDIDATE_BANK_TOP_K = 3
CANDIDATE_BANK_MAX_VARIANT_POSITIONS = 4
CANDIDATE_RERANKER_MODEL = "ridge"
CANDIDATE_RERANKER_MIN_CONTEXTS = 64
CANDIDATE_RERANKER_MIN_CANDIDATES = 256
CANDIDATE_CACHE_PREFIX = (
    f"{MODEL_CACHE_PREFIX}"
    f"_candidate_bank_{CANDIDATE_BANK_VERSION}"
    f"_topk{CANDIDATE_BANK_TOP_K}"
    f"_varpos{CANDIDATE_BANK_MAX_VARIANT_POSITIONS}"
    f"_reranker{CANDIDATE_RERANKER_MODEL}"
)
RERANKER_FEATURE_NAMES = (
    "ensemble_logprob_mean",
    "ensemble_candidate_prob_mean",
    "ensemble_margin_mean",
    "ensemble_entropy_mean",
    "fold_argmax_agreement_mean",
    "fold_candidate_prob_mean",
    "fold_candidate_prob_std_mean",
    "markov_logprob_mean",
    "position_prior_logprob_mean",
    "markov_match_ratio",
    "match_to_ensemble_direct_ratio",
    "adjacent_repeat_ratio",
    "unique_ratio",
    "max_run_ratio",
    "source_count",
    "source_is_ensemble_direct",
    "source_is_fold_direct",
    "source_is_fold_mix",
    "source_is_topk_variant",
    "source_is_markov_direct",
    "fold_count",
)


def resolve_transformer_query_config(
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    resolved_family = family_name or TRANSFORMER_QUERY_PRIMARY_FAMILY
    return dict(TRANSFORMER_QUERY_FAMILY_CONFIGS[resolved_family])


def build_model_cache_prefix(
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> str:
    resolved_config = resolve_transformer_query_config(
        config=config,
        family_name=family_name,
    )
    prefix = (
        f"{resolved_config['cache_prefix_root']}"
        f"_seed{RANDOM_STATE}"
        f"_maxlen{resolved_config['max_len']}"
        f"_dmodel{resolved_config['d_model']}"
        f"_nhead{resolved_config['nhead']}"
        f"_enc{resolved_config['encoder_layers']}"
        f"_dec{resolved_config['decoder_layers']}"
        f"_dropout{str(resolved_config['dropout']).replace('.', 'p')}"
        f"_epochs{resolved_config['epochs']}"
        f"_lr{str(resolved_config['lr']).replace('.', 'p')}"
        f"_wd{str(resolved_config['weight_decay']).replace('.', 'p')}"
        f"_scheduler{resolved_config['scheduler']}"
        f"_mask{int(resolved_config['use_padding_masks'])}"
    )
    synthetic_pretrain_epochs = int(
        resolved_config.get("synthetic_pretrain_epochs", 0)
    )
    synthetic_finetune_epochs = int(
        resolved_config.get("synthetic_finetune_epochs", 0)
    )
    if synthetic_pretrain_epochs > 0 or synthetic_finetune_epochs > 0:
        prefix += (
            f"_synthetic{synthetic_pretrain_epochs}"
            f"_finetune{synthetic_finetune_epochs}"
            f"_stride{int(resolved_config.get('synthetic_prefix_stride', 0))}"
            f"_minprefix{int(resolved_config.get('synthetic_min_prefix', 0))}"
        )
    return prefix


def uses_synthetic_prefix_augmentation(
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> bool:
    resolved_config = resolve_transformer_query_config(
        config=config,
        family_name=family_name,
    )
    return int(resolved_config.get("synthetic_pretrain_epochs", 0)) > 0


def build_synthetic_prefix_training_frame(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> pd.DataFrame:
    resolved_config = resolve_transformer_query_config(
        config=config,
        family_name=family_name,
    )
    stride = int(resolved_config.get("synthetic_prefix_stride", 0))
    min_prefix = int(resolved_config.get("synthetic_min_prefix", 0))
    if stride <= 0:
        return frame.reset_index(drop=True).copy()

    rows: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        full_sequence = clean_sequence(f"{row.context}{row.continuation}")
        max_prefix = len(full_sequence) - PREDICTION_LENGTH
        if max_prefix < min_prefix:
            continue

        prefix_positions = {len(clean_sequence(row.context))}
        for prefix_length in range(min_prefix, max_prefix + 1, stride):
            prefix_positions.add(prefix_length)

        for prefix_length in sorted(prefix_positions):
            if prefix_length < min_prefix:
                continue
            if prefix_length + PREDICTION_LENGTH > len(full_sequence):
                continue
            rows.append(
                {
                    "context": full_sequence[:prefix_length],
                    "continuation": full_sequence[
                        prefix_length : prefix_length + PREDICTION_LENGTH
                    ],
                }
            )

    if not rows:
        return frame.reset_index(drop=True).copy()
    return pd.DataFrame(rows)


def ensure_dirs() -> None:
    WORKING_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def torch_available() -> bool:
    return torch is not None and nn is not None and DataLoader is not None


def sklearn_available() -> bool:
    return (
        Ridge is not None
        and Pipeline is not None
        and StandardScaler is not None
    )


def set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)


def clean_sequence(raw: str) -> str:
    return "".join(
        character
        for character in str(raw).strip().upper()
        if character in ALPHABET_SET
    )


def score_predictions(predictions: list[str], targets: list[str]) -> float:
    correct = 0
    total = 0
    for prediction, target in zip(predictions, targets):
        for pred_char, target_char in zip(
            prediction[:PREDICTION_LENGTH],
            target[:PREDICTION_LENGTH],
        ):
            correct += pred_char == target_char
            total += 1
    return float(correct / total) if total else 0.0


def sequence_token_accuracy(prediction: str, target: str) -> float:
    return score_predictions([prediction], [target])


def select_device() -> str:
    if not torch_available():
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_markov_fallback(
    train_df: pd.DataFrame,
    order: int,
) -> tuple[list[dict[str, Counter]], Counter]:
    models: list[dict[str, Counter]] = [
        defaultdict(Counter) for _ in range(order + 1)
    ]
    global_counts: Counter = Counter()

    for row in train_df.itertuples(index=False):
        sequence = row.context + row.continuation
        if len(sequence) < 2:
            continue
        global_counts.update(sequence)
        for index in range(len(sequence) - 1):
            next_token = sequence[index + 1]
            for current_order in range(order + 1):
                start = max(0, index - current_order)
                context = sequence[start:index]
                models[current_order][context][next_token] += 1

    return models, global_counts


def score_sequence_with_markov(
    context: str,
    candidate: str,
    models: list[dict[str, Counter]],
    global_counts: Counter,
    order: int,
) -> float:
    history = context
    log_probability_sum = 0.0
    epsilon = 1e-8
    vocab_size = len(ALPHABET)
    default_total = sum(global_counts.values()) + vocab_size

    for token in candidate[:PREDICTION_LENGTH]:
        counter = None
        max_context = min(order, len(history))
        for current_order in range(max_context, -1, -1):
            key = history[-current_order:] if current_order > 0 else ""
            counter = models[current_order].get(key)
            if counter:
                break

        if counter:
            token_count = float(counter.get(token, 0)) + 1.0
            total = float(sum(counter.values())) + vocab_size
        else:
            token_count = float(global_counts.get(token, 0)) + 1.0
            total = float(default_total)

        log_probability_sum += math.log(max(token_count / total, epsilon))
        history += token

    return float(log_probability_sum / PREDICTION_LENGTH)


def predict_markov_fallback(
    context: str,
    models: list[dict[str, Counter]],
    global_counts: Counter,
    order: int,
) -> str:
    history = context
    generated: list[str] = []
    default_token = global_counts.most_common(1)[0][0]

    for _ in range(PREDICTION_LENGTH):
        chosen_token = None
        max_context = min(order, len(history))
        for current_order in range(max_context, -1, -1):
            key = history[-current_order:] if current_order > 0 else ""
            counter = models[current_order].get(key)
            if counter:
                chosen_token = counter.most_common(1)[0][0]
                break

        chosen_token = chosen_token or default_token
        generated.append(chosen_token)
        history += chosen_token

    return "".join(generated)


def encode_context(
    context: str,
    config: dict[str, Any] | None = None,
) -> list[int]:
    resolved_config = resolve_transformer_query_config(config=config)
    encoded = [CHAR_TO_INDEX[character] for character in context]
    encoded = encoded[-int(resolved_config["max_len"]) :]
    return encoded or [DEFAULT_CONTEXT_TOKEN]


def encode_target(continuation: str) -> list[int]:
    return [
        CHAR_TO_INDEX[character] - 1
        for character in continuation[:PREDICTION_LENGTH]
    ]


def normalize_candidate_sequence(sequence: str) -> str:
    cleaned = clean_sequence(sequence)
    if len(cleaned) >= PREDICTION_LENGTH:
        return cleaned[:PREDICTION_LENGTH]
    return cleaned + ("A" * (PREDICTION_LENGTH - len(cleaned)))


def sequence_to_target_indices(sequence: str) -> list[int]:
    normalized = normalize_candidate_sequence(sequence)
    return [CHAR_TO_INDEX[character] - 1 for character in normalized]


def get_train_valid_indices(
    frame: pd.DataFrame,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices, valid_indices = train_test_split(
        frame.index.to_numpy(),
        test_size=0.2,
        random_state=split_seed,
    )
    return np.asarray(train_indices), np.asarray(valid_indices)


def probabilities_to_sequence(probabilities: np.ndarray) -> str:
    token_indices = probabilities.argmax(axis=-1).tolist()
    return decode_target_indices(token_indices)


def sequence_match_ratio(left: str, right: str) -> float:
    return float(
        sum(
            left_char == right_char
            for left_char, right_char in zip(
                normalize_candidate_sequence(left),
                normalize_candidate_sequence(right),
            )
        )
        / PREDICTION_LENGTH
    )


def max_run_ratio(sequence: str) -> float:
    normalized = normalize_candidate_sequence(sequence)
    max_run = 1
    current_run = 1
    for index in range(1, len(normalized)):
        if normalized[index] == normalized[index - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return float(max_run / PREDICTION_LENGTH)


if torch_available():

    class TransformerQueryTrainDataset(Dataset):
        def __init__(
            self,
            frame: pd.DataFrame,
            config: dict[str, Any] | None = None,
        ) -> None:
            resolved_config = resolve_transformer_query_config(config=config)
            self.contexts = [
                torch.tensor(
                    encode_context(context, config=resolved_config),
                    dtype=torch.long,
                )
                for context in frame["context"].tolist()
            ]
            self.targets = [
                torch.tensor(
                    encode_target(continuation),
                    dtype=torch.long,
                )
                for continuation in frame["continuation"].tolist()
            ]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.contexts[index], self.targets[index]

    class TransformerQueryInferenceDataset(Dataset):
        def __init__(
            self,
            contexts: list[str],
            config: dict[str, Any] | None = None,
        ) -> None:
            resolved_config = resolve_transformer_query_config(config=config)
            self.contexts = [
                torch.tensor(
                    encode_context(context, config=resolved_config),
                    dtype=torch.long,
                )
                for context in contexts
            ]

        def __len__(self) -> int:
            return len(self.contexts)

        def __getitem__(self, index: int) -> torch.Tensor:
            return self.contexts[index]

    class TransformerQueryModel(nn.Module):
        def __init__(
            self,
            config: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self.config = resolve_transformer_query_config(config=config)
            self.token_embedding = nn.Embedding(
                len(CHAR_TO_INDEX) + 1,
                int(self.config["d_model"]),
                padding_idx=PAD_INDEX,
            )
            self.context_position_embedding = nn.Embedding(
                int(self.config["max_len"]),
                int(self.config["d_model"]),
            )
            self.query_embedding = nn.Parameter(
                torch.empty(
                    PREDICTION_LENGTH,
                    int(self.config["d_model"]),
                )
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(self.config["d_model"]),
                nhead=int(self.config["nhead"]),
                dim_feedforward=int(self.config["d_model"]) * 4,
                dropout=float(self.config["dropout"]),
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(self.config["encoder_layers"]),
                enable_nested_tensor=False,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=int(self.config["d_model"]),
                nhead=int(self.config["nhead"]),
                dim_feedforward=int(self.config["d_model"]) * 4,
                dropout=float(self.config["dropout"]),
                activation="gelu",
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=int(self.config["decoder_layers"]),
            )
            self.output = nn.Linear(
                int(self.config["d_model"]),
                len(CHAR_TO_INDEX),
            )
            nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)

        def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
            batch_size, context_length = context_tokens.shape
            context_positions = torch.arange(
                context_length,
                device=context_tokens.device,
            )
            context_embeddings = self.token_embedding(
                context_tokens
            ) + self.context_position_embedding(context_positions).unsqueeze(0)
            if bool(self.config["use_padding_masks"]):
                context_padding_mask = context_tokens.eq(PAD_INDEX)
                memory = self.encoder(
                    context_embeddings,
                    src_key_padding_mask=context_padding_mask,
                )
            else:
                memory = self.encoder(context_embeddings)

            query_tokens = self.query_embedding.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            if bool(self.config["use_padding_masks"]):
                decoded = self.decoder(
                    tgt=query_tokens,
                    memory=memory,
                    memory_key_padding_mask=context_padding_mask,
                )
            else:
                decoded = self.decoder(
                    tgt=query_tokens,
                    memory=memory,
                )
            return self.output(decoded)


def build_sequence_model(config: dict[str, Any] | None = None) -> Any:
    if not torch_available():
        return None
    return TransformerQueryModel(config=config)


def collate_transformer_query_train_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts, targets = zip(*batch)
    max_length = max(len(item) for item in contexts)
    context_batch = torch.full(
        (len(contexts), max_length),
        PAD_INDEX,
        dtype=torch.long,
    )

    for index, context in enumerate(contexts):
        context_batch[index, : len(context)] = context

    return context_batch, torch.stack(targets)


def collate_transformer_query_inference_batch(
    batch: list[torch.Tensor],
) -> torch.Tensor:
    max_length = max(len(item) for item in batch)
    context_batch = torch.full(
        (len(batch), max_length),
        PAD_INDEX,
        dtype=torch.long,
    )

    for index, context in enumerate(batch):
        context_batch[index, : len(context)] = context

    return context_batch


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def clear_device_cache(device: str) -> None:
    if not torch_available():
        return
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def decode_target_indices(token_indices: list[int]) -> str:
    return "".join(
        TARGET_TO_CHAR.get(int(token_index), "A")
        for token_index in token_indices[:PREDICTION_LENGTH]
    )


def get_model_cache_name(
    split_seed: int,
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> str:
    cache_prefix = build_model_cache_prefix(
        config=config,
        family_name=family_name,
    )
    return f"{cache_prefix}_split{split_seed}.pt"


def get_model_cache_path(
    split_seed: int,
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> Path:
    return CACHE_DIR / get_model_cache_name(
        split_seed,
        config=config,
        family_name=family_name,
    )


def get_candidate_bank_cache_name(split_seed: int) -> str:
    return f"{CANDIDATE_CACHE_PREFIX}_split{split_seed}.json"


def get_candidate_bank_cache_path(split_seed: int) -> Path:
    return CACHE_DIR / get_candidate_bank_cache_name(split_seed)


def get_reranker_cache_name() -> str:
    return f"{CANDIDATE_CACHE_PREFIX}_reranker.pkl"


def get_reranker_cache_path() -> Path:
    return CACHE_DIR / get_reranker_cache_name()


def get_reranker_meta_cache_name() -> str:
    return f"{CANDIDATE_CACHE_PREFIX}_reranker_meta.json"


def get_reranker_meta_cache_path() -> Path:
    return CACHE_DIR / get_reranker_meta_cache_name()


def get_inference_candidate_cache_name() -> str:
    return f"{CANDIDATE_CACHE_PREFIX}_test_candidates.json"


def get_inference_candidate_cache_path() -> Path:
    return CACHE_DIR / get_inference_candidate_cache_name()


def build_position_priors(train_df: pd.DataFrame) -> np.ndarray:
    priors = np.ones(
        (PREDICTION_LENGTH, len(ALPHABET)),
        dtype=np.float32,
    )
    for continuation in train_df["continuation"].tolist():
        for position_index, token_index in enumerate(
            encode_target(continuation)
        ):
            priors[position_index, token_index] += 1.0
    priors /= priors.sum(axis=1, keepdims=True)
    return priors


def build_reranker_pipeline() -> Any:
    if not sklearn_available():
        return None
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def feature_vector_from_mapping(features: dict[str, float]) -> list[float]:
    return [float(features[name]) for name in RERANKER_FEATURE_NAMES]


def normalize_probability_weights(
    weights: list[float] | tuple[float, ...] | np.ndarray | None,
    expected_count: int,
) -> np.ndarray:
    if expected_count <= 0:
        return np.zeros(0, dtype=np.float32)
    if weights is None:
        return np.full(expected_count, 1.0 / expected_count, dtype=np.float32)

    normalized = np.asarray(weights, dtype=np.float32).reshape(-1)
    if normalized.shape[0] != expected_count:
        return np.full(expected_count, 1.0 / expected_count, dtype=np.float32)

    normalized = np.clip(normalized, a_min=0.0, a_max=None)
    if not np.isfinite(normalized).all():
        return np.full(expected_count, 1.0 / expected_count, dtype=np.float32)

    weight_sum = float(normalized.sum())
    if weight_sum <= 0.0:
        return np.full(expected_count, 1.0 / expected_count, dtype=np.float32)
    return (normalized / weight_sum).astype(np.float32)


def get_fold_probability_weights(
    fold_metas: list[dict[str, Any]],
) -> list[float]:
    normalized = normalize_probability_weights(
        [float(meta.get("best_valid_score", 0.0)) for meta in fold_metas],
        expected_count=len(fold_metas),
    )
    return normalized.tolist()


def combine_model_probability_stack(
    stacked_probabilities: np.ndarray,
    model_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    if stacked_probabilities.ndim == 0:
        return stacked_probabilities.astype(np.float32)

    normalized_weights = normalize_probability_weights(
        model_weights,
        expected_count=int(stacked_probabilities.shape[0]),
    )
    return np.tensordot(
        normalized_weights,
        stacked_probabilities.astype(np.float32),
        axes=(0, 0),
    ).astype(np.float32)


def select_best_candidate_by_scores(
    candidate_rows: list[dict[str, Any]],
    predicted_scores: list[float],
) -> dict[str, Any]:
    best_index = max(
        range(len(candidate_rows)),
        key=lambda index: (
            float(predicted_scores[index]),
            -int(candidate_rows[index].get("candidate_rank", index)),
        ),
    )
    best_row = dict(candidate_rows[best_index])
    best_row["predicted_score"] = float(predicted_scores[best_index])
    return best_row


def summarize_group_selection_score(
    candidate_rows: list[dict[str, Any]],
    predicted_scores: list[float],
) -> float:
    grouped_rows: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(
        list
    )
    for row, predicted_score in zip(candidate_rows, predicted_scores):
        grouped_rows[str(row["group_id"])].append(
            (row, float(predicted_score))
        )

    chosen_scores: list[float] = []
    for grouped in grouped_rows.values():
        best_row, _ = max(
            grouped,
            key=lambda item: (
                item[1],
                -int(item[0].get("candidate_rank", 0)),
            ),
        )
        chosen_scores.append(float(best_row["target_score"]))
    return float(np.mean(chosen_scores)) if chosen_scores else 0.0


def summarize_group_reference_score(
    candidate_rows: list[dict[str, Any]],
    reducer: str,
) -> float:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        grouped_scores[str(row["group_id"])].append(float(row["target_score"]))

    if reducer == "oracle":
        values = [max(scores) for scores in grouped_scores.values()]
    else:
        baseline_scores = []
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            grouped_rows[str(row["group_id"])].append(row)
        for rows in grouped_rows.values():
            direct_row = min(
                rows,
                key=lambda item: int(item.get("candidate_rank", 0)),
            )
            baseline_scores.append(float(direct_row["target_score"]))
        values = baseline_scores

    return float(np.mean(values)) if values else 0.0


def score_transformer_query_model(
    model: Any,
    frame: pd.DataFrame,
    device: str,
    config: dict[str, Any] | None = None,
) -> float:
    if frame.empty:
        return 0.0

    batch_size = (
        TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED
        if device in {"cuda", "mps"}
        else TRANSFORMER_QUERY_BATCH_SIZE_CPU
    )
    dataset = TransformerQueryInferenceDataset(
        frame["context"].tolist(),
        config=config,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transformer_query_inference_batch,
        pin_memory=device == "cuda",
    )
    predictions: list[str] = []
    targets = frame["continuation"].tolist()

    model.eval()
    with torch.no_grad():
        for context_batch in loader:
            context_batch = context_batch.to(
                device,
                non_blocking=device == "cuda",
            )
            logits = model(context_batch)
            token_indices = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(
                decode_target_indices(row) for row in token_indices
            )

    return score_predictions(predictions, targets)


def predict_with_transformer_query_probability_matrices(
    contexts: list[str],
    model_states: list[dict[str, Any]],
    device: str,
    config: dict[str, Any] | None = None,
    model_weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not contexts:
        empty = np.zeros(
            (0, PREDICTION_LENGTH, len(ALPHABET)),
            dtype=np.float32,
        )
        return empty, []

    resolved_config = resolve_transformer_query_config(config=config)
    models: list[Any] = []
    for model_state in model_states:
        model = build_sequence_model(config=resolved_config)
        if model is None:
            empty = np.zeros(
                (len(contexts), PREDICTION_LENGTH, len(ALPHABET)),
                dtype=np.float32,
            )
            return empty, []
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        models.append(model)

    batch_size = (
        TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED
        if device in {"cuda", "mps"}
        else TRANSFORMER_QUERY_BATCH_SIZE_CPU
    )
    dataset = TransformerQueryInferenceDataset(
        contexts,
        config=resolved_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transformer_query_inference_batch,
        pin_memory=device == "cuda",
    )
    mean_batches: list[np.ndarray] = []
    per_model_batches: list[list[np.ndarray]] = [[] for _ in models]

    with torch.no_grad():
        for context_batch in loader:
            context_batch = context_batch.to(
                device,
                non_blocking=device == "cuda",
            )
            model_probabilities: list[np.ndarray] = []
            for model in models:
                logits = model(context_batch)
                probabilities = torch.softmax(logits, dim=-1)
                model_probabilities.append(
                    probabilities.detach().cpu().numpy().astype(np.float32)
                )

            stacked_probabilities = np.stack(model_probabilities, axis=0)
            mean_batches.append(
                combine_model_probability_stack(
                    stacked_probabilities,
                    model_weights=model_weights,
                )
            )
            for model_index, probabilities in enumerate(model_probabilities):
                per_model_batches[model_index].append(probabilities)

    clear_device_cache(device)

    mean_probabilities = np.concatenate(mean_batches, axis=0)
    per_model_probabilities = [
        np.concatenate(probability_chunks, axis=0)
        for probability_chunks in per_model_batches
    ]
    return mean_probabilities, per_model_probabilities


def train_transformer_query_fold_model(
    train_df: pd.DataFrame,
    device: str,
    split_seed: int,
    config: dict[str, Any] | None = None,
    family_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_config = resolve_transformer_query_config(
        config=config,
        family_name=family_name,
    )
    cache_path = get_model_cache_path(split_seed, config=resolved_config)
    cache_enabled = os.environ.get("ERIS_ALLOW_CACHE", "1") == "1"
    if cache_enabled and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["model_state"], {
            "cache_hit": True,
            "epoch_losses": payload.get("epoch_losses", []),
            "epochs": payload.get("epochs", int(resolved_config["epochs"])),
            "batch_size": payload.get(
                "batch_size",
                TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED,
            ),
            "scheduler": payload.get(
                "scheduler",
                str(resolved_config["scheduler"]),
            ),
            "best_epoch": payload.get(
                "best_epoch",
                int(resolved_config["epochs"]),
            ),
            "best_valid_score": payload.get("best_valid_score", 0.0),
            "split_seed": payload.get("split_seed", split_seed),
            "family_name": payload.get(
                "family_name",
                str(resolved_config["family_name"]),
            ),
        }

    train_fold, valid_fold = train_test_split(
        train_df,
        test_size=0.2,
        random_state=split_seed,
    )
    train_fold = train_fold.reset_index(drop=True)
    valid_fold = valid_fold.reset_index(drop=True)
    set_seeds(RANDOM_STATE)

    batch_size = (
        TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED
        if device in {"cuda", "mps"}
        else TRANSFORMER_QUERY_BATCH_SIZE_CPU
    )
    train_loaders: list[tuple[str, Any]] = []
    if uses_synthetic_prefix_augmentation(config=resolved_config):
        synthetic_pretrain_epochs = int(
            resolved_config.get("synthetic_pretrain_epochs", 0)
        )
        synthetic_train_frame = build_synthetic_prefix_training_frame(
            train_fold,
            config=resolved_config,
        )
        synthetic_train_dataset = TransformerQueryTrainDataset(
            synthetic_train_frame,
            config=resolved_config,
        )
        synthetic_train_loader = DataLoader(
            synthetic_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_transformer_query_train_batch,
            pin_memory=device == "cuda",
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        train_loaders.extend(
            [("synthetic_pretrain", synthetic_train_loader)]
            * synthetic_pretrain_epochs
        )
    else:
        synthetic_pretrain_epochs = 0

    finetune_epochs = int(
        resolved_config.get(
            "synthetic_finetune_epochs",
            max(int(resolved_config["epochs"]) - synthetic_pretrain_epochs, 1),
        )
    )
    train_dataset = TransformerQueryTrainDataset(
        train_fold,
        config=resolved_config,
    )
    finetune_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_transformer_query_train_batch,
        pin_memory=device == "cuda",
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    train_loaders.extend([("finetune", finetune_loader)] * finetune_epochs)

    valid_dataset = TransformerQueryInferenceDataset(
        valid_fold["context"].tolist(),
        config=resolved_config,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_transformer_query_inference_batch,
        pin_memory=device == "cuda",
    )
    valid_targets = valid_fold["continuation"].tolist()

    model = build_sequence_model(config=resolved_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved_config["lr"]),
        weight_decay=float(resolved_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(len(train_loaders), 1),
    )
    criterion = nn.CrossEntropyLoss()
    use_cuda_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    epoch_losses: list[float] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_valid_score = -1.0

    for epoch_index, (_, train_loader) in enumerate(train_loaders, start=1):
        model.train()
        losses: list[float] = []
        for context_batch, target_batch in train_loader:
            context_batch = context_batch.to(
                device,
                non_blocking=device == "cuda",
            )
            target_batch = target_batch.to(
                device,
                non_blocking=device == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)
            if use_cuda_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    logits = model(context_batch)
                    loss = criterion(
                        logits.transpose(1, 2),
                        target_batch,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(resolved_config["grad_clip"]),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(context_batch)
                loss = criterion(
                    logits.transpose(1, 2),
                    target_batch,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(resolved_config["grad_clip"]),
                )
                optimizer.step()
            losses.append(float(loss.item()))

        epoch_losses.append(float(np.mean(losses)))
        scheduler.step()
        valid_predictions: list[str] = []
        model.eval()
        with torch.no_grad():
            for context_batch in valid_loader:
                context_batch = context_batch.to(
                    device,
                    non_blocking=device == "cuda",
                )
                logits = model(context_batch)
                token_indices = logits.argmax(dim=-1).cpu().tolist()
                valid_predictions.extend(
                    decode_target_indices(row) for row in token_indices
                )

        valid_score = score_predictions(valid_predictions, valid_targets)
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_epoch = epoch_index
            best_state = clone_state_dict(model)

    model_state = best_state or clone_state_dict(model)
    if cache_enabled:
        torch.save(
            {
                "model_state": model_state,
                "epoch_losses": epoch_losses,
                "epochs": int(resolved_config["epochs"]),
                "batch_size": batch_size,
                "scheduler": str(resolved_config["scheduler"]),
                "model_family": "transformer_query",
                "best_epoch": best_epoch,
                "best_valid_score": best_valid_score,
                "split_seed": split_seed,
                "family_name": str(resolved_config["family_name"]),
            },
            cache_path,
        )

    clear_device_cache(device)

    return model_state, {
        "cache_hit": False,
        "epoch_losses": epoch_losses,
        "epochs": int(resolved_config["epochs"]),
        "batch_size": batch_size,
        "scheduler": str(resolved_config["scheduler"]),
        "best_epoch": best_epoch,
        "best_valid_score": best_valid_score,
        "split_seed": split_seed,
        "family_name": str(resolved_config["family_name"]),
    }


def predict_with_transformer_query(
    contexts: list[str],
    model_state: dict[str, Any],
    device: str,
    config: dict[str, Any] | None = None,
) -> list[str]:
    mean_probabilities, _ = (
        predict_with_transformer_query_probability_matrices(
            contexts=contexts,
            model_states=[model_state],
            device=device,
            config=config,
        )
    )
    if mean_probabilities.size == 0 and contexts:
        return ["A" * PREDICTION_LENGTH] * len(contexts)
    return [
        probabilities_to_sequence(probabilities)
        for probabilities in mean_probabilities
    ]


def predict_with_transformer_query_ensemble(
    contexts: list[str],
    model_states: list[dict[str, Any]],
    device: str,
    config: dict[str, Any] | None = None,
) -> list[str]:
    mean_probabilities, _ = (
        predict_with_transformer_query_probability_matrices(
            contexts=contexts,
            model_states=model_states,
            device=device,
            config=config,
        )
    )
    if mean_probabilities.size == 0 and contexts:
        return ["A" * PREDICTION_LENGTH] * len(contexts)
    return [
        probabilities_to_sequence(probabilities)
        for probabilities in mean_probabilities
    ]


def train_transformer_query_family(
    train_df: pd.DataFrame,
    device: str,
    family_name: str,
) -> dict[str, Any]:
    config = resolve_transformer_query_config(family_name=family_name)
    split_seeds = tuple(
        TRANSFORMER_QUERY_FAMILY_SPLIT_SEEDS.get(
            family_name,
            TRANSFORMER_QUERY_ENSEMBLE_SPLIT_SEEDS,
        )
    )
    model_states: list[dict[str, Any]] = []
    fold_metas: list[dict[str, Any]] = []
    for split_seed in split_seeds:
        model_state, training_meta = train_transformer_query_fold_model(
            train_df,
            device=device,
            split_seed=split_seed,
            config=config,
        )
        model_states.append(model_state)
        fold_metas.append(training_meta)

    return {
        "family_name": family_name,
        "config": config,
        "weight": float(TRANSFORMER_QUERY_FAMILY_WEIGHTS[family_name]),
        "split_seeds": split_seeds,
        "model_states": model_states,
        "fold_metas": fold_metas,
        "fold_probability_weights": get_fold_probability_weights(fold_metas),
    }


def combine_weighted_probability_groups(
    probability_groups: dict[str, np.ndarray],
    family_weights: dict[str, float] | None = None,
) -> np.ndarray:
    if not probability_groups:
        return np.zeros(
            (0, PREDICTION_LENGTH, len(ALPHABET)),
            dtype=np.float32,
        )

    resolved_weights = family_weights or TRANSFORMER_QUERY_FAMILY_WEIGHTS
    weighted_sum = None
    total_weight = 0.0
    for family_name in TRANSFORMER_QUERY_FAMILY_ORDER:
        probabilities = probability_groups.get(family_name)
        if probabilities is None:
            continue
        family_weight = float(resolved_weights.get(family_name, 0.0))
        if weighted_sum is None:
            weighted_sum = probabilities.astype(np.float32) * family_weight
        else:
            weighted_sum = weighted_sum + (
                probabilities.astype(np.float32) * family_weight
            )
        total_weight += family_weight

    if weighted_sum is None or total_weight <= 0.0:
        first_group = next(iter(probability_groups.values()))
        return first_group.astype(np.float32)
    return (weighted_sum / total_weight).astype(np.float32)


def resolve_length_gated_family_weights(
    context_length: int,
    available_family_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, float]:
    available = set(available_family_names)
    if context_length > 320 or 224 < context_length <= 256:
        desired_weights = {"v8": 1.0}
    else:
        desired_weights = {"v10": 0.8, "v11b": 0.2}

    filtered_weights = {
        family_name: weight
        for family_name, weight in desired_weights.items()
        if family_name in available
    }
    if filtered_weights:
        normalized = normalize_probability_weights(
            list(filtered_weights.values()),
            expected_count=len(filtered_weights),
        )
        return {
            family_name: float(weight)
            for family_name, weight in zip(
                filtered_weights.keys(),
                normalized.tolist(),
            )
        }

    fallback_families = [
        family_name
        for family_name in TRANSFORMER_QUERY_FAMILY_ORDER
        if family_name in available
    ]
    if not fallback_families:
        return {}
    fallback_weights = normalize_probability_weights(
        [
            float(TRANSFORMER_QUERY_FAMILY_WEIGHTS.get(family_name, 1.0))
            for family_name in fallback_families
        ],
        expected_count=len(fallback_families),
    )
    return {
        family_name: float(weight)
        for family_name, weight in zip(
            fallback_families,
            fallback_weights.tolist(),
        )
    }


def combine_length_gated_probability_groups(
    contexts: list[str],
    family_probability_groups: dict[str, np.ndarray],
) -> np.ndarray:
    if not contexts or not family_probability_groups:
        return np.zeros(
            (0, PREDICTION_LENGTH, len(ALPHABET)),
            dtype=np.float32,
        )

    available_family_names = [
        family_name
        for family_name, probabilities in family_probability_groups.items()
        if probabilities.size > 0
    ]
    reference_probabilities = next(iter(family_probability_groups.values()))
    combined_rows: list[np.ndarray] = []
    for row_index, context in enumerate(contexts):
        family_weights = resolve_length_gated_family_weights(
            context_length=len(context),
            available_family_names=available_family_names,
        )
        if not family_weights:
            combined_rows.append(reference_probabilities[row_index])
            continue

        weighted_row = None
        total_weight = 0.0
        for family_name, family_weight in family_weights.items():
            row_probabilities = family_probability_groups[family_name][
                row_index
            ]
            if weighted_row is None:
                weighted_row = (
                    row_probabilities.astype(np.float32) * family_weight
                )
            else:
                weighted_row = weighted_row + (
                    row_probabilities.astype(np.float32) * family_weight
                )
            total_weight += float(family_weight)

        if weighted_row is None or total_weight <= 0.0:
            combined_rows.append(reference_probabilities[row_index])
        else:
            combined_rows.append(
                (weighted_row / total_weight).astype(np.float32)
            )

    return np.stack(combined_rows, axis=0).astype(np.float32)


def predict_with_heterogeneous_transformer_query_probability_matrices(
    contexts: list[str],
    family_payloads: list[dict[str, Any]],
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    family_probability_groups: dict[str, np.ndarray] = {}
    for family_payload in family_payloads:
        family_name = str(family_payload["family_name"])
        family_mean_probabilities, _ = (
            predict_with_transformer_query_probability_matrices(
                contexts=contexts,
                model_states=family_payload["model_states"],
                device=device,
                config=family_payload["config"],
                model_weights=family_payload.get("fold_probability_weights"),
            )
        )
        family_probability_groups[family_name] = family_mean_probabilities

    combined_probabilities = combine_length_gated_probability_groups(
        contexts=contexts,
        family_probability_groups=family_probability_groups,
    )
    return combined_probabilities, family_probability_groups


def predict_with_heterogeneous_transformer_query_ensemble(
    contexts: list[str],
    family_payloads: list[dict[str, Any]],
    device: str,
) -> list[str]:
    combined_probabilities, _ = (
        predict_with_heterogeneous_transformer_query_probability_matrices(
            contexts=contexts,
            family_payloads=family_payloads,
            device=device,
        )
    )
    if combined_probabilities.size == 0 and contexts:
        return ["A" * PREDICTION_LENGTH] * len(contexts)
    return [
        probabilities_to_sequence(probabilities)
        for probabilities in combined_probabilities
    ]


def build_candidate_bank_for_context(
    context: str,
    mean_probabilities: np.ndarray,
    per_model_probabilities: list[np.ndarray],
    markov_models: list[dict[str, Counter]],
    global_counts: Counter,
    position_priors: np.ndarray,
) -> list[dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}
    ensemble_direct = probabilities_to_sequence(mean_probabilities)
    markov_direct = predict_markov_fallback(
        context=context,
        models=markov_models,
        global_counts=global_counts,
        order=MARKOV_FALLBACK_ORDER,
    )
    fold_direct_sequences = [
        probabilities_to_sequence(probabilities)
        for probabilities in per_model_probabilities
    ]

    source_priority = {
        "ensemble_direct": 0,
        "fold_direct": 1,
        "fold_mix": 2,
        "topk_variant": 3,
        "markov_blend": 4,
        "markov_direct": 5,
    }

    def register_candidate(candidate: str, source: str) -> None:
        normalized = normalize_candidate_sequence(candidate)
        record = candidate_map.setdefault(
            normalized,
            {
                "candidate": normalized,
                "sources": [],
            },
        )
        if source not in record["sources"]:
            record["sources"].append(source)

    register_candidate(ensemble_direct, "ensemble_direct")
    register_candidate(markov_direct, "markov_direct")
    for fold_direct in fold_direct_sequences:
        register_candidate(fold_direct, "fold_direct")

    if fold_direct_sequences:
        mixed_tokens: list[str] = []
        for position_index in range(PREDICTION_LENGTH):
            vote_counter = Counter(
                sequence[position_index] for sequence in fold_direct_sequences
            )
            vote_counter[ensemble_direct[position_index]] += 1e-3
            mixed_tokens.append(vote_counter.most_common(1)[0][0])
        fold_mix_candidate = "".join(mixed_tokens)
        register_candidate(fold_mix_candidate, "fold_mix")

    sorted_indices = np.argsort(mean_probabilities, axis=-1)
    top1_indices = sorted_indices[:, -1]
    topk_indices = sorted_indices[:, -CANDIDATE_BANK_TOP_K:][:, ::-1]
    top1_probabilities = mean_probabilities[
        np.arange(PREDICTION_LENGTH),
        top1_indices,
    ]
    top2_indices = topk_indices[:, 1]
    top2_probabilities = mean_probabilities[
        np.arange(PREDICTION_LENGTH),
        top2_indices,
    ]
    uncertainty_positions = np.argsort(
        top1_probabilities - top2_probabilities
    )[:CANDIDATE_BANK_MAX_VARIANT_POSITIONS]

    for position_index in uncertainty_positions.tolist():
        for alternative_rank in range(1, CANDIDATE_BANK_TOP_K):
            alternative_index = int(
                topk_indices[position_index, alternative_rank]
            )
            alternative_character = TARGET_TO_CHAR[alternative_index]
            if alternative_character == ensemble_direct[position_index]:
                continue
            candidate_tokens = list(ensemble_direct)
            candidate_tokens[position_index] = alternative_character
            register_candidate("".join(candidate_tokens), "topk_variant")

    blended_tokens = list(ensemble_direct)
    blend_changed = False
    for position_index in uncertainty_positions.tolist():
        if markov_direct[position_index] != ensemble_direct[position_index]:
            blended_tokens[position_index] = markov_direct[position_index]
            blend_changed = True
    if blend_changed:
        register_candidate("".join(blended_tokens), "markov_blend")

    mean_entropy = -(
        mean_probabilities * np.log(np.clip(mean_probabilities, 1e-8, 1.0))
    ).sum(axis=-1)
    fold_candidate_probabilities = []
    fold_direct_agreements = []

    candidates = sorted(
        candidate_map.values(),
        key=lambda item: min(
            source_priority[source] for source in item["sources"]
        ),
    )

    for candidate_rank, candidate_row in enumerate(candidates):
        candidate = candidate_row["candidate"]
        candidate_indices = np.asarray(sequence_to_target_indices(candidate))
        candidate_probabilities = mean_probabilities[
            np.arange(PREDICTION_LENGTH),
            candidate_indices,
        ]
        fold_candidate_probabilities.clear()
        fold_direct_agreements.clear()
        for fold_probabilities in per_model_probabilities:
            fold_candidate_probabilities.append(
                fold_probabilities[
                    np.arange(PREDICTION_LENGTH),
                    candidate_indices,
                ]
            )
            fold_direct_agreements.append(
                sequence_match_ratio(
                    candidate,
                    probabilities_to_sequence(fold_probabilities),
                )
            )

        if fold_candidate_probabilities:
            stacked_fold_probabilities = np.stack(
                fold_candidate_probabilities,
                axis=0,
            )
            fold_candidate_prob_mean = float(stacked_fold_probabilities.mean())
            fold_candidate_prob_std_mean = float(
                stacked_fold_probabilities.std(axis=0).mean()
            )
            fold_argmax_agreement_mean = float(np.mean(fold_direct_agreements))
        else:
            fold_candidate_prob_mean = 0.0
            fold_candidate_prob_std_mean = 0.0
            fold_argmax_agreement_mean = 0.0

        primary_source = min(
            candidate_row["sources"],
            key=lambda source: source_priority[source],
        )
        unique_ratio = float(len(set(candidate)) / PREDICTION_LENGTH)
        adjacent_repeat_ratio = float(
            sum(
                candidate[index] == candidate[index - 1]
                for index in range(1, PREDICTION_LENGTH)
            )
            / max(PREDICTION_LENGTH - 1, 1)
        )
        features = {
            "ensemble_logprob_mean": float(
                np.log(np.clip(candidate_probabilities, 1e-8, 1.0)).mean()
            ),
            "ensemble_candidate_prob_mean": float(
                candidate_probabilities.mean()
            ),
            "ensemble_margin_mean": float(
                (top1_probabilities - top2_probabilities).mean()
            ),
            "ensemble_entropy_mean": float(mean_entropy.mean()),
            "fold_argmax_agreement_mean": fold_argmax_agreement_mean,
            "fold_candidate_prob_mean": fold_candidate_prob_mean,
            "fold_candidate_prob_std_mean": (fold_candidate_prob_std_mean),
            "markov_logprob_mean": score_sequence_with_markov(
                context=context,
                candidate=candidate,
                models=markov_models,
                global_counts=global_counts,
                order=MARKOV_FALLBACK_ORDER,
            ),
            "position_prior_logprob_mean": float(
                np.log(
                    np.clip(
                        position_priors[
                            np.arange(PREDICTION_LENGTH),
                            candidate_indices,
                        ],
                        1e-8,
                        1.0,
                    )
                ).mean()
            ),
            "markov_match_ratio": sequence_match_ratio(
                candidate,
                markov_direct,
            ),
            "match_to_ensemble_direct_ratio": sequence_match_ratio(
                candidate,
                ensemble_direct,
            ),
            "adjacent_repeat_ratio": adjacent_repeat_ratio,
            "unique_ratio": unique_ratio,
            "max_run_ratio": max_run_ratio(candidate),
            "source_count": float(len(candidate_row["sources"])),
            "source_is_ensemble_direct": float(
                "ensemble_direct" in candidate_row["sources"]
            ),
            "source_is_fold_direct": float(
                "fold_direct" in candidate_row["sources"]
            ),
            "source_is_fold_mix": float(
                "fold_mix" in candidate_row["sources"]
            ),
            "source_is_topk_variant": float(
                "topk_variant" in candidate_row["sources"]
            ),
            "source_is_markov_direct": float(
                "markov_direct" in candidate_row["sources"]
            ),
            "fold_count": float(len(per_model_probabilities)),
        }
        candidate_row["primary_source"] = primary_source
        candidate_row["candidate_rank"] = candidate_rank
        candidate_row["feature_mapping"] = features
        candidate_row["feature_vector"] = feature_vector_from_mapping(features)

    return candidates


def build_oof_candidate_rows(
    train_df: pd.DataFrame,
    model_states: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[bool]]:
    cache_enabled = os.environ.get("ERIS_ALLOW_CACHE", "1") == "1"
    candidate_rows: list[dict[str, Any]] = []
    cache_hits: list[bool] = []

    for split_seed, model_state in zip(
        TRANSFORMER_QUERY_ENSEMBLE_SPLIT_SEEDS,
        model_states,
    ):
        cache_path = get_candidate_bank_cache_path(split_seed)
        if cache_enabled and cache_path.exists():
            candidate_rows.extend(
                json.loads(cache_path.read_text(encoding="utf-8"))
            )
            cache_hits.append(True)
            continue

        train_indices, valid_indices = get_train_valid_indices(
            train_df,
            split_seed=split_seed,
        )
        train_fold = train_df.loc[train_indices].reset_index(drop=True)
        valid_fold = train_df.loc[valid_indices].reset_index(drop=True)
        markov_models, global_counts = build_markov_fallback(
            train_fold,
            order=MARKOV_FALLBACK_ORDER,
        )
        position_priors = build_position_priors(train_fold)
        mean_probabilities, per_model_probabilities = (
            predict_with_transformer_query_probability_matrices(
                contexts=valid_fold["context"].tolist(),
                model_states=[model_state],
                device=device,
            )
        )
        split_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(valid_fold.itertuples(index=False)):
            candidate_bank = build_candidate_bank_for_context(
                context=row.context,
                mean_probabilities=mean_probabilities[row_index],
                per_model_probabilities=[
                    probabilities[row_index]
                    for probabilities in per_model_probabilities
                ],
                markov_models=markov_models,
                global_counts=global_counts,
                position_priors=position_priors,
            )
            for candidate_row in candidate_bank:
                split_rows.append(
                    {
                        "group_id": f"{split_seed}:{row_index}",
                        "split_seed": int(split_seed),
                        "context": row.context,
                        "target": row.continuation,
                        "candidate": candidate_row["candidate"],
                        "sources": candidate_row["sources"],
                        "primary_source": (candidate_row["primary_source"]),
                        "candidate_rank": int(candidate_row["candidate_rank"]),
                        "feature_vector": [
                            float(value)
                            for value in candidate_row["feature_vector"]
                        ],
                        "target_score": sequence_token_accuracy(
                            candidate_row["candidate"],
                            row.continuation,
                        ),
                    }
                )
        if cache_enabled:
            cache_path.write_text(
                json.dumps(split_rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        candidate_rows.extend(split_rows)
        cache_hits.append(False)

    return candidate_rows, cache_hits


def train_candidate_reranker(
    candidate_rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    cache_enabled = os.environ.get("ERIS_ALLOW_CACHE", "1") == "1"
    cache_path = get_reranker_cache_path()
    meta_path = get_reranker_meta_cache_path()
    if cache_enabled and cache_path.exists() and meta_path.exists():
        with cache_path.open("rb") as handle:
            model = pickle.load(handle)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["cache_hit"] = True
        return model, meta

    group_count = len({str(row["group_id"]) for row in candidate_rows})
    if not sklearn_available():
        raise RuntimeError("sklearn no disponible para el reranker")
    if group_count < CANDIDATE_RERANKER_MIN_CONTEXTS:
        raise RuntimeError("candidate bank demasiado pequeno para reranker")
    if len(candidate_rows) < CANDIDATE_RERANKER_MIN_CANDIDATES:
        raise RuntimeError("pocos candidatos para entrenar reranker")

    features = np.asarray(
        [row["feature_vector"] for row in candidate_rows],
        dtype=np.float32,
    )
    targets = np.asarray(
        [float(row["target_score"]) for row in candidate_rows],
        dtype=np.float32,
    )
    groups = np.asarray(
        [str(row["group_id"]) for row in candidate_rows],
        dtype=object,
    )
    split_count = min(5, group_count)
    oof_predictions = np.zeros(len(candidate_rows), dtype=np.float32)

    if split_count >= 2:
        group_kfold = GroupKFold(n_splits=split_count)
        for train_index, valid_index in group_kfold.split(
            features,
            targets,
            groups,
        ):
            model = build_reranker_pipeline()
            if model is None:
                raise RuntimeError("no se pudo construir el reranker")
            model.fit(features[train_index], targets[train_index])
            oof_predictions[valid_index] = model.predict(
                features[valid_index]
            ).astype(np.float32)
    else:
        oof_predictions[:] = targets

    final_model = build_reranker_pipeline()
    if final_model is None:
        raise RuntimeError("no se pudo construir el reranker final")
    final_model.fit(features, targets)

    meta = {
        "cache_hit": False,
        "group_count": int(group_count),
        "candidate_rows": int(len(candidate_rows)),
        "baseline_score": round(
            summarize_group_reference_score(
                candidate_rows, reducer="baseline"
            ),
            6,
        ),
        "oof_selected_score": round(
            summarize_group_selection_score(
                candidate_rows,
                oof_predictions.tolist(),
            ),
            6,
        ),
        "oracle_score": round(
            summarize_group_reference_score(candidate_rows, reducer="oracle"),
            6,
        ),
        "feature_names": list(RERANKER_FEATURE_NAMES),
        "model_name": CANDIDATE_RERANKER_MODEL,
    }
    if cache_enabled:
        with cache_path.open("wb") as handle:
            pickle.dump(final_model, handle)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return final_model, meta


def rerank_test_candidates(
    test_df: pd.DataFrame,
    mean_probabilities: np.ndarray,
    per_model_probabilities: list[np.ndarray],
    markov_models: list[dict[str, Counter]],
    global_counts: Counter,
    position_priors: np.ndarray,
    reranker_model: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    predictions: list[str] = []
    payload: list[dict[str, Any]] = []

    for row_index, row in enumerate(test_df.itertuples(index=False)):
        candidate_rows = build_candidate_bank_for_context(
            context=row.context,
            mean_probabilities=mean_probabilities[row_index],
            per_model_probabilities=[
                probabilities[row_index]
                for probabilities in per_model_probabilities
            ],
            markov_models=markov_models,
            global_counts=global_counts,
            position_priors=position_priors,
        )
        feature_matrix = np.asarray(
            [row_item["feature_vector"] for row_item in candidate_rows],
            dtype=np.float32,
        )
        predicted_scores = reranker_model.predict(feature_matrix).tolist()
        best_row = select_best_candidate_by_scores(
            candidate_rows,
            predicted_scores,
        )
        predictions.append(best_row["candidate"])
        payload.append(
            {
                "seq_id": row.seq_id,
                "selected": best_row["candidate"],
                "predicted_score": round(
                    float(best_row["predicted_score"]),
                    6,
                ),
                "primary_source": best_row["primary_source"],
                "sources": best_row["sources"],
                "candidate_count": len(candidate_rows),
            }
        )

    return predictions, payload


def main() -> None:
    ensure_dirs()
    set_seeds()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_df["context"] = train_df["context"].map(clean_sequence)
    train_df["continuation"] = train_df["continuation"].map(clean_sequence)
    test_df["context"] = test_df["context"].map(clean_sequence)

    strategy = "markov_fallback"
    device = "cpu"
    epoch_losses: list[float] = []
    cache_hit = False
    training_seconds = 0.0
    inference_seconds = 0.0
    runtime_error = ""
    fold_metas: list[dict[str, Any]] = []
    family_payloads: list[dict[str, Any]] = []
    family_probability_groups: dict[str, np.ndarray] = {}
    family_runtime_errors: dict[str, str] = {}

    try:
        if torch_available():
            strategy = "transformer_query_heterogeneous_ensemble"
            device = select_device()

            training_start = time.perf_counter()
            for family_name in TRANSFORMER_QUERY_FAMILY_ORDER:
                try:
                    family_payload = train_transformer_query_family(
                        train_df=train_df,
                        device=device,
                        family_name=family_name,
                    )
                except Exception as family_exc:  # pragma: no cover
                    family_runtime_errors[family_name] = str(family_exc)
                    continue
                family_payloads.append(family_payload)
                fold_metas.extend(family_payload["fold_metas"])
            if not family_payloads:
                raise RuntimeError(
                    "ninguna familia transformer disponible para inferencia"
                )
            training_seconds = time.perf_counter() - training_start
            epoch_losses = [
                float(np.mean(meta["epoch_losses"]))
                for meta in fold_metas
                if meta["epoch_losses"]
            ]
            cache_hit = all(bool(meta["cache_hit"]) for meta in fold_metas)

            inference_start = time.perf_counter()
            mean_probabilities, family_probability_groups = (
                predict_with_heterogeneous_transformer_query_probability_matrices(
                    contexts=test_df["context"].tolist(),
                    family_payloads=family_payloads,
                    device=device,
                )
            )
            predictions = [
                probabilities_to_sequence(probabilities)
                for probabilities in mean_probabilities
            ]
            get_inference_candidate_cache_path().unlink(missing_ok=True)
            inference_seconds = time.perf_counter() - inference_start
        else:
            raise RuntimeError("torch no disponible")
    except Exception as exc:  # pragma: no cover - fallback defensivo
        runtime_error = str(exc)
        strategy = "markov_fallback"
        device = "cpu"

        training_start = time.perf_counter()
        models, global_counts = build_markov_fallback(
            train_df,
            order=MARKOV_FALLBACK_ORDER,
        )
        training_seconds = time.perf_counter() - training_start

        inference_start = time.perf_counter()
        predictions = [
            predict_markov_fallback(
                context,
                models,
                global_counts,
                order=MARKOV_FALLBACK_ORDER,
            )
            for context in test_df["context"]
        ]
        inference_seconds = time.perf_counter() - inference_start

    submission = pd.DataFrame(
        {
            "seq_id": test_df["seq_id"],
            "continuation": predictions,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)

    report = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "prediction_length": PREDICTION_LENGTH,
        "strategy": strategy,
        "device": device,
        "torch_available": torch_available(),
        "sklearn_available": sklearn_available(),
        "model_family": (
            "transformer_query_heterogeneous_ensemble"
            if strategy == "transformer_query_heterogeneous_ensemble"
            else "markov_fallback"
        ),
        "transformer_query_max_len": TRANSFORMER_QUERY_MAX_LEN,
        "transformer_query_d_model": TRANSFORMER_QUERY_D_MODEL,
        "transformer_query_nhead": TRANSFORMER_QUERY_NHEAD,
        "transformer_query_encoder_layers": (TRANSFORMER_QUERY_ENCODER_LAYERS),
        "transformer_query_decoder_layers": (TRANSFORMER_QUERY_DECODER_LAYERS),
        "transformer_query_dropout": TRANSFORMER_QUERY_DROPOUT,
        "transformer_query_epochs": TRANSFORMER_QUERY_EPOCHS,
        "transformer_query_batch_size": (
            TRANSFORMER_QUERY_BATCH_SIZE_ACCELERATED
            if device in {"cuda", "mps"}
            else TRANSFORMER_QUERY_BATCH_SIZE_CPU
        ),
        "transformer_query_lr": TRANSFORMER_QUERY_LR,
        "transformer_query_weight_decay": (TRANSFORMER_QUERY_WEIGHT_DECAY),
        "transformer_query_scheduler": TRANSFORMER_QUERY_SCHEDULER,
        "transformer_query_use_padding_masks": (
            TRANSFORMER_QUERY_USE_PADDING_MASKS
        ),
        "transformer_query_validation_metric": "exact_token_accuracy",
        "transformer_query_validation_decode": "direct_argmax",
        "transformer_query_test_decode": (
            "length_gated_family_softmax_mean_argmax"
            if strategy == "transformer_query_heterogeneous_ensemble"
            else "softmax_mean_argmax"
        ),
        "transformer_query_target_indexing": "0_based",
        "transformer_query_split_seeds": list(
            TRANSFORMER_QUERY_ENSEMBLE_SPLIT_SEEDS
        ),
        "transformer_query_primary_family": (TRANSFORMER_QUERY_PRIMARY_FAMILY),
        "transformer_query_family_order": list(TRANSFORMER_QUERY_FAMILY_ORDER),
        "transformer_query_family_weights": {
            family_name: float(TRANSFORMER_QUERY_FAMILY_WEIGHTS[family_name])
            for family_name in TRANSFORMER_QUERY_FAMILY_ORDER
        },
        "transformer_query_family_fold_weighting": (
            "best_valid_score_proportional"
        ),
        "transformer_query_length_gating": {
            "long_context_rules": [
                {
                    "context_len_gt": 320,
                    "weights": {"v8": 1.0},
                },
                {
                    "context_len_gt": 224,
                    "context_len_lte": 256,
                    "weights": {"v8": 1.0},
                },
            ],
            "default_weights": {"v10": 0.8, "v11b": 0.2},
            "fallback_order": list(TRANSFORMER_QUERY_FAMILY_ORDER),
        },
        "transformer_query_family_configs": {
            family_name: resolve_transformer_query_config(
                family_name=family_name
            )
            for family_name in TRANSFORMER_QUERY_FAMILY_ORDER
        },
        "transformer_query_fold_epoch_losses": {
            f"{meta['family_name']}:{meta['split_seed']}": meta["epoch_losses"]
            for meta in fold_metas
        },
        "transformer_query_fold_best_epochs": {
            f"{meta['family_name']}:{meta['split_seed']}": int(
                meta["best_epoch"]
            )
            for meta in fold_metas
        },
        "transformer_query_fold_best_valid_scores": {
            f"{meta['family_name']}:{meta['split_seed']}": round(
                float(meta["best_valid_score"]),
                6,
            )
            for meta in fold_metas
        },
        "transformer_query_epoch_losses": epoch_losses,
        "model_cache_names": [
            get_model_cache_name(split_seed)
            for split_seed in TRANSFORMER_QUERY_ENSEMBLE_SPLIT_SEEDS
        ],
        "transformer_query_family_model_cache_names": {
            family_name: [
                get_model_cache_name(
                    split_seed,
                    family_name=family_name,
                )
                for split_seed in TRANSFORMER_QUERY_FAMILY_SPLIT_SEEDS[
                    family_name
                ]
            ]
            for family_name in TRANSFORMER_QUERY_FAMILY_ORDER
        },
        "transformer_query_family_cache_hits": {
            family_payload["family_name"]: all(
                bool(meta["cache_hit"])
                for meta in family_payload["fold_metas"]
            )
            for family_payload in family_payloads
        },
        "transformer_query_family_fold_probability_weights": {
            family_payload["family_name"]: [
                round(float(weight), 6)
                for weight in family_payload.get(
                    "fold_probability_weights",
                    [],
                )
            ]
            for family_payload in family_payloads
        },
        "transformer_query_family_fold_best_valid_score_mean": {
            family_payload["family_name"]: round(
                float(
                    np.mean(
                        [
                            meta["best_valid_score"]
                            for meta in family_payload["fold_metas"]
                        ]
                    )
                ),
                6,
            )
            for family_payload in family_payloads
        },
        "transformer_query_heterogeneous_family_probability_shapes": {
            family_name: list(probabilities.shape)
            for family_name, probabilities in family_probability_groups.items()
        },
        "transformer_query_family_runtime_errors": family_runtime_errors,
        "candidate_reranker_enabled": False,
        "cache_hit": cache_hit,
        "markov_fallback_order": MARKOV_FALLBACK_ORDER,
        "training_seconds": round(training_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "runtime_error": runtime_error,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
