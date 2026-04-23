from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


DATA_DIR = "./dataset/public"
WORKING_DIR = "./working"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(WORKING_DIR, "submission.csv")

RANDOM_SEED = 42
CATBOOST_SEEDS = [11, 42, 73]
VARIABILITY_SPECS = [
    (12000, 18000, 4.0),
    (16000, 22000, 3.0),
    (10000, 16000, 6.0),
]
FOLLOWUP_TEXT_SPEC = (18000, 26000, 4.0)
FOLLOWUP_STRUCT_WEIGHT = 0.88
VARIABILITY_FALLBACK_THRESHOLD = 0.48
DISTANCE_EXPLICIT_TREE_BLEND = 0.75
ENABLE_STUDENT_BRANCH = False

CLEAN_TARGETS = [
    "transient_class",
    "host_environment",
    "spectral_regime",
]
SCORED_TARGETS = [
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
    "precursor_category",
]
ALL_TARGETS = CLEAN_TARGETS + SCORED_TARGETS

BASE_STRUCT_FEATURES = [
    "transient_class",
    "host_environment",
    "spectral_regime",
    "class_group",
    "env_zone",
    "z_num",
    "L_num",
    "z_missing",
    "L_missing",
]
DISTANCE_FEATURES = [
    "transient_class",
    "host_environment",
    "spectral_regime",
    "class_group",
    "env_zone",
    "z_num",
    "z_missing",
]
ENERGY_FEATURES = [
    "transient_class",
    "spectral_regime",
    "class_group",
    "env_zone",
    "L_num",
    "L_missing",
]
FOLLOWUP_FEATURES = [
    "transient_class",
    "spectral_regime",
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "class_group",
    "env_zone",
]

DISTANCE_TREE_FEATURE = ["z_num"]

Z_PATTERN = re.compile(r"\bz\s*=\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
L_PATTERN_PRIMARY = re.compile(
    r"(?:peak\s+)?log(?:arithmic)?\s*(?:luminosity(?:\s*\(L\))?|\(?\s*L\s*\)?)"
    r"(?:\s+value)?\s*(?:of|=)?\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE,
)
L_PATTERN_POWER = re.compile(
    r"luminosity[^\n]{0,80}10\^\{([0-9]+\.[0-9]+)\}",
    re.IGNORECASE,
)

DISTANCE_PRIOR_BLEND = 0.6

STUDENT_TARGETS = [
    "transient_class",
    "host_environment",
    "spectral_regime",
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
]
STUDENT_HEAD_WEIGHTS = {
    "transient_class": 0.4,
    "host_environment": 0.4,
    "spectral_regime": 0.4,
    "variability_pattern": 1.0,
    "distance_bin": 0.8,
    "energy_tier": 0.8,
    "followup_protocol": 1.0,
}
STUDENT_LABEL_SMOOTHING = {
    "transient_class": 0.0,
    "host_environment": 0.0,
    "spectral_regime": 0.0,
    "variability_pattern": 0.05,
    "distance_bin": 0.05,
    "energy_tier": 0.05,
    "followup_protocol": 0.05,
}
STUDENT_BLEND_WEIGHTS = {
    "variability_pattern": 0.25,
    "followup_protocol": 0.35,
}


def mode_value(series: pd.Series) -> str:
    return series.value_counts().idxmax()


def add_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["z_num"] = frame["narrative"].str.extract(Z_PATTERN)[0].astype(float)
    luminosity_primary = frame["narrative"].str.extract(L_PATTERN_PRIMARY)[0]
    luminosity_power = frame["narrative"].str.extract(L_PATTERN_POWER)[0]
    frame["L_num"] = luminosity_primary.fillna(luminosity_power).astype(float)
    frame["z_missing"] = frame["z_num"].isna().astype(int)
    frame["L_missing"] = frame["L_num"].isna().astype(int)
    return frame


def add_group_features(
    frame: pd.DataFrame,
    class_group: dict[str, int],
    env_zone: dict[str, int],
) -> pd.DataFrame:
    frame = frame.copy()
    frame["class_group"] = (
        frame["transient_class"].map(class_group).fillna(-1).astype(int).astype(str)
    )
    frame["env_zone"] = (
        frame["host_environment"].map(env_zone).fillna(-1).astype(int).astype(str)
    )
    return frame


def build_sparse_text_features(
    train_text: pd.Series,
    infer_text: pd.Series,
    *,
    word_max_features: int,
    char_max_features: int,
) -> tuple[TfidfVectorizer, TfidfVectorizer, object, object]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=word_max_features,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=char_max_features,
        sublinear_tf=True,
    )
    x_train = hstack(
        [
            word_vectorizer.fit_transform(train_text),
            char_vectorizer.fit_transform(train_text),
        ]
    ).tocsr()
    x_infer = hstack(
        [
            word_vectorizer.transform(infer_text),
            char_vectorizer.transform(infer_text),
        ]
    ).tocsr()
    return word_vectorizer, char_vectorizer, x_train, x_infer


@dataclass
class CleanModelBundle:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    models: dict[str, LinearSVC]


def fit_clean_models(train_df: pd.DataFrame) -> CleanModelBundle:
    word_vectorizer, char_vectorizer, x_train, _ = build_sparse_text_features(
        train_df["narrative"],
        train_df["narrative"],
        word_max_features=20000,
        char_max_features=30000,
    )
    models: dict[str, LinearSVC] = {}
    for target in CLEAN_TARGETS:
        model = LinearSVC(C=1.0)
        model.fit(x_train, train_df[target])
        models[target] = model
    return CleanModelBundle(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        models=models,
    )


def predict_clean_labels(bundle: CleanModelBundle, infer_df: pd.DataFrame) -> pd.DataFrame:
    x_infer = hstack(
        [
            bundle.word_vectorizer.transform(infer_df["narrative"]),
            bundle.char_vectorizer.transform(infer_df["narrative"]),
        ]
    ).tocsr()
    predictions = {}
    for target, model in bundle.models.items():
        predictions[target] = model.predict(x_infer)
    return pd.DataFrame(predictions, index=infer_df.index)


@dataclass
class VariabilityBundle:
    members: list[tuple[TfidfVectorizer, TfidfVectorizer, LogisticRegression]]
    classes: np.ndarray
    fallback_map: dict[tuple[str, str], str]
    threshold: float


def fit_variability_model(train_df: pd.DataFrame) -> VariabilityBundle:
    members: list[tuple[TfidfVectorizer, TfidfVectorizer, LogisticRegression]] = []
    for word_max_features, char_max_features, c_value in VARIABILITY_SPECS:
        word_vectorizer, char_vectorizer, x_train, _ = build_sparse_text_features(
            train_df["narrative"],
            train_df["narrative"],
            word_max_features=word_max_features,
            char_max_features=char_max_features,
        )
        model = LogisticRegression(
            max_iter=1400,
            C=c_value,
            n_jobs=1,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train, train_df["variability_pattern"])
        members.append((word_vectorizer, char_vectorizer, model))
    fallback_map = (
        train_df.groupby(["transient_class", "spectral_regime"])["variability_pattern"]
        .agg(mode_value)
        .to_dict()
    )
    return VariabilityBundle(
        members=members,
        classes=np.asarray(sorted(train_df["variability_pattern"].unique())),
        fallback_map=fallback_map,
        threshold=VARIABILITY_FALLBACK_THRESHOLD,
    )


def predict_variability_probabilities(
    bundle: VariabilityBundle,
    infer_df: pd.DataFrame,
) -> np.ndarray:
    probabilities = np.zeros((len(infer_df), len(bundle.classes)), dtype=float)
    for word_vectorizer, char_vectorizer, model in bundle.members:
        x_infer = hstack(
            [
                word_vectorizer.transform(infer_df["narrative"]),
                char_vectorizer.transform(infer_df["narrative"]),
            ]
        ).tocsr()
        model_probabilities = model.predict_proba(x_infer)
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        order = [class_to_index[label] for label in bundle.classes]
        probabilities += model_probabilities[:, order]
    probabilities /= len(bundle.members)
    return probabilities


def predict_variability(
    bundle: VariabilityBundle,
    infer_df: pd.DataFrame,
    clean_predictions: pd.DataFrame,
) -> np.ndarray:
    probabilities = predict_variability_probabilities(bundle, infer_df)
    predicted = bundle.classes[probabilities.argmax(axis=1)]

    low_confidence = probabilities.max(axis=1) < bundle.threshold
    low_indices = np.where(low_confidence)[0]
    for idx in low_indices:
        key = (
            clean_predictions.iloc[idx]["transient_class"],
            clean_predictions.iloc[idx]["spectral_regime"],
        )
        fallback = bundle.fallback_map.get(key)
        if fallback is not None:
            predicted[idx] = fallback
    return predicted


def predict_variability_with_probabilities(
    bundle: VariabilityBundle,
    infer_df: pd.DataFrame,
    clean_predictions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = predict_variability_probabilities(bundle, infer_df)
    predicted = bundle.classes[probabilities.argmax(axis=1)]

    low_confidence = probabilities.max(axis=1) < bundle.threshold
    for idx in np.where(low_confidence)[0]:
        key = (
            clean_predictions.iloc[idx]["transient_class"],
            clean_predictions.iloc[idx]["spectral_regime"],
        )
        fallback = bundle.fallback_map.get(key)
        if fallback is not None:
            predicted[idx] = fallback
            probabilities[idx] = 0.0
            probabilities[idx, np.where(bundle.classes == fallback)[0][0]] = 1.0
    return predicted, probabilities


def make_catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        depth=6,
        learning_rate=0.08,
        iterations=300,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )


@dataclass
class StructuredModelBundle:
    distance_models: list[CatBoostClassifier]
    distance_classes: np.ndarray
    distance_tree: DecisionTreeClassifier
    distance_prior_map: dict[tuple[str, str], np.ndarray]
    distance_global_prior: np.ndarray
    energy_models: list[CatBoostClassifier]
    energy_classes: np.ndarray
    followup_models: list[CatBoostClassifier]
    followup_classes: np.ndarray
    followup_text_word_vectorizer: TfidfVectorizer
    followup_text_char_vectorizer: TfidfVectorizer
    followup_text_model: LogisticRegression
    precursor_class_group: dict[str, int]
    precursor_env_zone: dict[str, int]
    precursor_block_majority: dict[tuple[int, int], str]
    precursor_pair_mode: dict[tuple[str, str], str]
    precursor_global_majority: str


def build_precursor_groups(
    train_df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[tuple[int, int], str], str]:
    pair_mode = (
        train_df.groupby(["transient_class", "host_environment"])["precursor_category"]
        .agg(mode_value)
        .unstack()
    )
    categories = sorted(train_df["precursor_category"].unique())

    class_features = []
    for _, row in pair_mode.iterrows():
        feature = [(row == category).sum() for category in categories]
        for environment in pair_mode.columns:
            for category in categories:
                feature.append(1 if row[environment] == category else 0)
        class_features.append(feature)
    class_features = np.asarray(class_features, dtype=float)

    env_features = []
    for environment in pair_mode.columns:
        column = pair_mode[environment]
        feature = [(column == category).sum() for category in categories]
        for transient_class in pair_mode.index:
            for category in categories:
                feature.append(1 if column[transient_class] == category else 0)
        env_features.append(feature)
    env_features = np.asarray(env_features, dtype=float)

    class_labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(class_features)
    env_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(env_features)

    class_group = dict(zip(pair_mode.index, class_labels))
    env_zone = dict(zip(pair_mode.columns, env_labels))

    block_votes: dict[tuple[int, int], list[str]] = defaultdict(list)
    for transient_class in pair_mode.index:
        for environment in pair_mode.columns:
            precursor = pair_mode.loc[transient_class, environment]
            if pd.notna(precursor):
                block_votes[(class_group[transient_class], env_zone[environment])].append(precursor)
    block_majority = {
        block: Counter(values).most_common(1)[0][0]
        for block, values in block_votes.items()
    }
    global_majority = mode_value(train_df["precursor_category"])
    return class_group, env_zone, block_majority, global_majority


def fit_structured_models(train_df: pd.DataFrame) -> StructuredModelBundle:
    (
        precursor_class_group,
        precursor_env_zone,
        precursor_block_majority,
        precursor_global_majority,
    ) = build_precursor_groups(train_df)
    grouped_train = add_group_features(
        train_df,
        precursor_class_group,
        precursor_env_zone,
    )

    distance_models = []
    for seed in CATBOOST_SEEDS:
        distance_model = make_catboost()
        distance_model.set_params(random_seed=seed)
        distance_model.fit(
            grouped_train[DISTANCE_FEATURES],
            grouped_train["distance_bin"],
            cat_features=[0, 1, 2, 3, 4],
        )
        distance_models.append(distance_model)

    distance_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=30,
        random_state=RANDOM_SEED,
    )
    distance_tree.fit(
        grouped_train.loc[grouped_train["z_num"].notna(), DISTANCE_TREE_FEATURE],
        grouped_train.loc[grouped_train["z_num"].notna(), "distance_bin"],
    )

    distance_prior_source = grouped_train.loc[grouped_train["z_num"].notna()]
    distance_global_prior = (
        distance_prior_source["distance_bin"]
        .value_counts(normalize=True)
        .reindex(sorted(grouped_train["distance_bin"].unique()), fill_value=0.0)
        .to_numpy(dtype=float)
    )
    distance_prior_map = {}
    for key, subset in distance_prior_source.groupby(["transient_class", "spectral_regime"]):
        prior = (
            subset["distance_bin"]
            .value_counts(normalize=True)
            .reindex(sorted(grouped_train["distance_bin"].unique()), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        distance_prior_map[key] = prior

    energy_models = []
    for seed in CATBOOST_SEEDS:
        energy_model = make_catboost()
        energy_model.set_params(random_seed=seed)
        energy_model.fit(
            grouped_train[ENERGY_FEATURES],
            grouped_train["energy_tier"],
            cat_features=[0, 1, 2, 3],
        )
        energy_models.append(energy_model)

    followup_models = []
    for seed in CATBOOST_SEEDS:
        followup_model = make_catboost()
        followup_model.set_params(random_seed=seed)
        followup_model.fit(
            grouped_train[FOLLOWUP_FEATURES],
            grouped_train["followup_protocol"],
            cat_features=[0, 1, 2, 3, 4, 5, 6],
        )
        followup_models.append(followup_model)

    (
        followup_text_word_vectorizer,
        followup_text_char_vectorizer,
        followup_x_train,
        _,
    ) = build_sparse_text_features(
        train_df["narrative"],
        train_df["narrative"],
        word_max_features=FOLLOWUP_TEXT_SPEC[0],
        char_max_features=FOLLOWUP_TEXT_SPEC[1],
    )
    followup_text_model = LogisticRegression(
        max_iter=1500,
        C=FOLLOWUP_TEXT_SPEC[2],
        n_jobs=1,
        random_state=RANDOM_SEED,
    )
    followup_text_model.fit(followup_x_train, train_df["followup_protocol"])

    precursor_pair_mode = (
        train_df.groupby(["transient_class", "host_environment"])["precursor_category"]
        .agg(mode_value)
        .to_dict()
    )

    return StructuredModelBundle(
        distance_models=distance_models,
        distance_classes=np.asarray(sorted(train_df["distance_bin"].unique())),
        distance_tree=distance_tree,
        distance_prior_map=distance_prior_map,
        distance_global_prior=distance_global_prior,
        energy_models=energy_models,
        energy_classes=np.asarray(sorted(train_df["energy_tier"].unique())),
        followup_models=followup_models,
        followup_classes=np.asarray(sorted(train_df["followup_protocol"].unique())),
        followup_text_word_vectorizer=followup_text_word_vectorizer,
        followup_text_char_vectorizer=followup_text_char_vectorizer,
        followup_text_model=followup_text_model,
        precursor_class_group=precursor_class_group,
        precursor_env_zone=precursor_env_zone,
        precursor_block_majority=precursor_block_majority,
        precursor_pair_mode=precursor_pair_mode,
        precursor_global_majority=precursor_global_majority,
    )


def predict_distance(
    bundle: StructuredModelBundle,
    structured_df: pd.DataFrame,
) -> np.ndarray:
    grouped_df = add_group_features(
        structured_df,
        bundle.precursor_class_group,
        bundle.precursor_env_zone,
    )
    probabilities = np.zeros((len(grouped_df), len(bundle.distance_classes)), dtype=float)
    for model in bundle.distance_models:
        model_probabilities = model.predict_proba(grouped_df[DISTANCE_FEATURES])
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        order = [class_to_index[label] for label in bundle.distance_classes]
        probabilities += model_probabilities[:, order]
    probabilities /= len(bundle.distance_models)
    missing_rows = grouped_df["z_num"].isna().values
    if missing_rows.any():
        for idx in np.where(missing_rows)[0]:
            key = (
                grouped_df.iloc[idx]["transient_class"],
                grouped_df.iloc[idx]["spectral_regime"],
            )
            prior = bundle.distance_prior_map.get(key, bundle.distance_global_prior)
            probabilities[idx] = (
                DISTANCE_PRIOR_BLEND * probabilities[idx]
                + (1.0 - DISTANCE_PRIOR_BLEND) * prior
            )
    explicit_rows = grouped_df["z_num"].notna().values
    if explicit_rows.any():
        tree_probabilities = bundle.distance_tree.predict_proba(
            grouped_df.loc[explicit_rows, DISTANCE_TREE_FEATURE]
        )
        class_to_index = {
            label: idx for idx, label in enumerate(bundle.distance_tree.classes_)
        }
        order = [class_to_index[label] for label in bundle.distance_classes]
        probabilities[explicit_rows] = (
            (1.0 - DISTANCE_EXPLICIT_TREE_BLEND) * probabilities[explicit_rows]
            + DISTANCE_EXPLICIT_TREE_BLEND * tree_probabilities[:, order]
        )
    predicted = bundle.distance_classes[probabilities.argmax(axis=1)]
    return predicted


def predict_energy(
    bundle: StructuredModelBundle,
    structured_df: pd.DataFrame,
) -> np.ndarray:
    grouped_df = add_group_features(
        structured_df,
        bundle.precursor_class_group,
        bundle.precursor_env_zone,
    )
    probabilities = np.zeros((len(grouped_df), len(bundle.energy_classes)), dtype=float)
    for model in bundle.energy_models:
        model_probabilities = model.predict_proba(grouped_df[ENERGY_FEATURES])
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        order = [class_to_index[label] for label in bundle.energy_classes]
        probabilities += model_probabilities[:, order]
    probabilities /= len(bundle.energy_models)
    return bundle.energy_classes[probabilities.argmax(axis=1)]


def predict_followup(
    bundle: StructuredModelBundle,
    structured_df: pd.DataFrame,
) -> np.ndarray:
    probabilities = predict_followup_probabilities(bundle, structured_df)
    return bundle.followup_classes[probabilities.argmax(axis=1)]


def predict_followup_probabilities(
    bundle: StructuredModelBundle,
    structured_df: pd.DataFrame,
) -> np.ndarray:
    grouped_df = add_group_features(
        structured_df,
        bundle.precursor_class_group,
        bundle.precursor_env_zone,
    )
    structured_probabilities = np.zeros((len(grouped_df), len(bundle.followup_classes)), dtype=float)
    for model in bundle.followup_models:
        model_probabilities = model.predict_proba(grouped_df[FOLLOWUP_FEATURES])
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        order = [class_to_index[label] for label in bundle.followup_classes]
        structured_probabilities += model_probabilities[:, order]
    structured_probabilities /= len(bundle.followup_models)

    text_features = hstack(
        [
            bundle.followup_text_word_vectorizer.transform(grouped_df["narrative"]),
            bundle.followup_text_char_vectorizer.transform(grouped_df["narrative"]),
        ]
    ).tocsr()
    text_probabilities = bundle.followup_text_model.predict_proba(text_features)
    class_to_index = {
        label: idx for idx, label in enumerate(bundle.followup_text_model.classes_)
    }
    order = [class_to_index[label] for label in bundle.followup_classes]
    text_probabilities = text_probabilities[:, order]

    probabilities = (
        FOLLOWUP_STRUCT_WEIGHT * structured_probabilities
        + (1.0 - FOLLOWUP_STRUCT_WEIGHT) * text_probabilities
    )
    return probabilities


def predict_followup_with_probabilities(
    bundle: StructuredModelBundle,
    structured_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = predict_followup_probabilities(bundle, structured_df)
    return bundle.followup_classes[probabilities.argmax(axis=1)], probabilities


def predict_precursor(
    bundle: StructuredModelBundle,
    clean_predictions: pd.DataFrame,
) -> np.ndarray:
    predicted = []
    for _, row in clean_predictions.iterrows():
        pair = (row["transient_class"], row["host_environment"])
        if pair in bundle.precursor_pair_mode:
            predicted.append(bundle.precursor_pair_mode[pair])
            continue
        class_group = bundle.precursor_class_group.get(row["transient_class"])
        env_zone = bundle.precursor_env_zone.get(row["host_environment"])
        precursor = bundle.precursor_block_majority.get(
            (class_group, env_zone),
            bundle.precursor_global_majority,
        )
        predicted.append(precursor)
    return np.asarray(predicted)


if TORCH_AVAILABLE:

    @dataclass
    class StudentConfig:
        model_name: str
        epochs: int
        learning_rate: float
        train_batch_size: int
        infer_batch_size: int
        gradient_accumulation: int
        max_length: int
        pseudo_weight: float


    class StudentDataset(Dataset):
        def __init__(
            self,
            encodings: dict[str, torch.Tensor],
            numeric_features: np.ndarray,
            labels: dict[str, np.ndarray],
            sample_weights: np.ndarray,
        ) -> None:
            self.encodings = encodings
            self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
            self.labels = {
                target: torch.tensor(values, dtype=torch.long)
                for target, values in labels.items()
            }
            self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        def __len__(self) -> int:
            return self.numeric_features.shape[0]

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            item = {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "numeric_features": self.numeric_features[idx],
                "sample_weight": self.sample_weights[idx],
            }
            for target, values in self.labels.items():
                item[f"label_{target}"] = values[idx]
            return item


    class StudentMultiTaskModel(nn.Module):
        def __init__(
            self,
            model_name: str,
            head_sizes: dict[str, int],
            numeric_dim: int,
            *,
            local_files_only: bool,
        ) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            hidden_size = self.backbone.config.hidden_size
            self.numeric_proj = nn.Sequential(
                nn.Linear(numeric_dim, 32),
                nn.GELU(),
                nn.LayerNorm(32),
                nn.Dropout(0.1),
            )
            self.dropout = nn.Dropout(0.1)
            self.heads = nn.ModuleDict(
                {
                    target: nn.Linear(hidden_size + 32, size)
                    for target, size in head_sizes.items()
                }
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            numeric_features: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            fused = torch.cat([pooled, self.numeric_proj(numeric_features)], dim=1)
            fused = self.dropout(fused)
            return {
                target: head(fused)
                for target, head in self.heads.items()
            }


def detect_student_device() -> tuple[object | None, float]:
    if not TORCH_AVAILABLE:
        return None, 0.0
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return torch.device("cuda"), total_gb
    return torch.device("cpu"), 0.0


def pick_student_config(total_gpu_gb: float) -> StudentConfig | None:
    if not TORCH_AVAILABLE:
        return None
    model_override = os.environ.get("ERIS_STUDENT_MODEL")
    epochs_override = os.environ.get("ERIS_STUDENT_EPOCHS")
    batch_override = os.environ.get("ERIS_STUDENT_BATCH")
    accum_override = os.environ.get("ERIS_STUDENT_ACCUM")
    infer_batch_override = os.environ.get("ERIS_STUDENT_INFER_BATCH")
    if model_override:
        config = StudentConfig(
            model_name=model_override,
            epochs=int(epochs_override or 2),
            learning_rate=1.5e-5 if "base" in model_override else 2e-5,
            train_batch_size=int(batch_override or 4),
            infer_batch_size=int(infer_batch_override or 12),
            gradient_accumulation=int(accum_override or 2),
            max_length=192,
            pseudo_weight=0.2,
        )
        return config
    if total_gpu_gb >= 20:
        return StudentConfig(
            model_name="microsoft/deberta-v3-base",
            epochs=3,
            learning_rate=1.5e-5,
            train_batch_size=12,
            infer_batch_size=24,
            gradient_accumulation=1,
            max_length=192,
            pseudo_weight=0.2,
        )
    if total_gpu_gb >= 10:
        return StudentConfig(
            model_name="microsoft/deberta-v3-base",
            epochs=2,
            learning_rate=1.5e-5,
            train_batch_size=6,
            infer_batch_size=16,
            gradient_accumulation=2,
            max_length=192,
            pseudo_weight=0.2,
        )
    if total_gpu_gb >= 6:
        return StudentConfig(
            model_name="microsoft/deberta-v3-small",
            epochs=1,
            learning_rate=2e-5,
            train_batch_size=4,
            infer_batch_size=12,
            gradient_accumulation=2,
            max_length=192,
            pseudo_weight=0.15,
        )
    return None


def build_student_numeric_features(frame: pd.DataFrame) -> np.ndarray:
    z_filled = frame["z_num"].fillna(0.0).to_numpy(dtype=np.float32)
    l_filled = frame["L_num"].fillna(0.0).to_numpy(dtype=np.float32)
    numeric = np.column_stack(
        [
            z_filled,
            l_filled,
            frame["z_missing"].to_numpy(dtype=np.float32),
            frame["L_missing"].to_numpy(dtype=np.float32),
        ]
    )
    numeric[:, 0] /= 10.0
    numeric[:, 1] /= 50.0
    return numeric


def fit_student_transformer(
    train_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
) -> dict[str, np.ndarray] | None:
    device, total_gpu_gb = detect_student_device()
    config = pick_student_config(total_gpu_gb)
    if not TORCH_AVAILABLE or device is None or config is None or device.type != "cuda":
        return None

    tokenizer = None
    model = None
    for local_only in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                local_files_only=local_only,
            )
            model = StudentMultiTaskModel(
                config.model_name,
                head_sizes={
                    target: train_df[target].nunique()
                    for target in STUDENT_TARGETS
                },
                numeric_dim=4,
                local_files_only=local_only,
            ).to(device)
            break
        except Exception:
            tokenizer = None
            model = None
    if tokenizer is None or model is None:
        return None

    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        label_maps = {
            target: {
                label: idx
                for idx, label in enumerate(sorted(train_df[target].unique()))
            }
            for target in STUDENT_TARGETS
        }
        combined_df = pd.concat([train_df, pseudo_df], ignore_index=True)
        sample_weights = np.concatenate(
            [
                np.ones(len(train_df), dtype=np.float32),
                np.full(len(pseudo_df), config.pseudo_weight, dtype=np.float32),
            ]
        )
        encodings = tokenizer(
            combined_df["narrative"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt",
        )
        labels = {
            target: combined_df[target].map(label_maps[target]).to_numpy(dtype=np.int64)
            for target in STUDENT_TARGETS
        }
        dataset = StudentDataset(
            encodings=encodings,
            numeric_features=build_student_numeric_features(combined_df),
            labels=labels,
            sample_weights=sample_weights,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

        total_steps = max(
            1,
            int(np.ceil(len(train_loader) * config.epochs / config.gradient_accumulation)),
        )
        warmup_steps = max(1, int(total_steps * 0.06))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=not torch.cuda.is_bf16_supported())
        use_autocast = device.type == "cuda"
        use_bf16 = torch.cuda.is_bf16_supported()
        total_head_weight = sum(STUDENT_HEAD_WEIGHTS[target] for target in STUDENT_TARGETS)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(config.epochs):
            for step, batch in enumerate(train_loader, 1):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with torch.autocast(
                    device_type=device.type,
                    enabled=use_autocast,
                    dtype=autocast_dtype,
                ):
                    logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        numeric_features=batch["numeric_features"],
                    )
                    per_sample_loss = torch.zeros(
                        batch["input_ids"].size(0),
                        device=device,
                        dtype=torch.float32,
                    )
                    for target in STUDENT_TARGETS:
                        head_loss = F.cross_entropy(
                            logits[target],
                            batch[f"label_{target}"],
                            reduction="none",
                            label_smoothing=STUDENT_LABEL_SMOOTHING[target],
                        )
                        per_sample_loss = per_sample_loss + STUDENT_HEAD_WEIGHTS[target] * head_loss
                    loss = (
                        (per_sample_loss * batch["sample_weight"]).mean()
                        / total_head_weight
                        / config.gradient_accumulation
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % config.gradient_accumulation == 0 or step == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        model.eval()
        test_encodings = tokenizer(
            pseudo_df["narrative"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt",
        )
        test_dataset = StudentDataset(
            encodings=test_encodings,
            numeric_features=build_student_numeric_features(pseudo_df),
            labels={
                target: pseudo_df[target].map(label_maps[target]).to_numpy(dtype=np.int64)
                for target in STUDENT_TARGETS
            },
            sample_weights=np.ones(len(pseudo_df), dtype=np.float32),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.infer_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        probabilities = {
            target: np.zeros((len(pseudo_df), len(label_maps[target])), dtype=np.float32)
            for target in STUDENT_TARGETS
        }
        offset = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numeric_features=batch["numeric_features"],
                )
                batch_size = batch["input_ids"].size(0)
                for target in STUDENT_TARGETS:
                    probs = torch.softmax(logits[target], dim=1).detach().cpu().numpy()
                    probabilities[target][offset : offset + batch_size] = probs
                offset += batch_size

        torch.cuda.empty_cache()
        return probabilities
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count does not match sample_submission.csv")

    valid_values = {
        column: set(sample_submission[column].unique())
        for column in ALL_TARGETS
    }
    train_df = pd.read_csv(TRAIN_PATH)
    for column in ALL_TARGETS:
        valid_values[column].update(train_df[column].unique())
        invalid = set(submission[column].unique()) - valid_values[column]
        if invalid:
            raise ValueError(f"Invalid values detected in {column}: {sorted(invalid)}")


def main() -> None:
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_df = add_numeric_features(pd.read_csv(TRAIN_PATH))
    test_df = add_numeric_features(pd.read_csv(TEST_PATH))
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    clean_bundle = fit_clean_models(train_df)
    clean_predictions = predict_clean_labels(clean_bundle, test_df)

    variability_bundle = fit_variability_model(train_df)
    variability_predictions, variability_probabilities = predict_variability_with_probabilities(
        variability_bundle,
        test_df,
        clean_predictions,
    )

    structured_bundle = fit_structured_models(train_df)

    structured_test = test_df.copy()
    for column in CLEAN_TARGETS:
        structured_test[column] = clean_predictions[column].values

    distance_predictions = predict_distance(structured_bundle, structured_test)
    energy_predictions = predict_energy(structured_bundle, structured_test)

    structured_test["variability_pattern"] = variability_predictions
    structured_test["distance_bin"] = distance_predictions
    structured_test["energy_tier"] = energy_predictions

    followup_predictions, followup_probabilities = predict_followup_with_probabilities(
        structured_bundle,
        structured_test,
    )
    precursor_predictions = predict_precursor(structured_bundle, clean_predictions)

    student_probabilities = None
    if ENABLE_STUDENT_BRANCH:
        teacher_pseudo_df = test_df.copy()
        for column in CLEAN_TARGETS:
            teacher_pseudo_df[column] = clean_predictions[column].values
        teacher_pseudo_df["variability_pattern"] = variability_predictions
        teacher_pseudo_df["distance_bin"] = distance_predictions
        teacher_pseudo_df["energy_tier"] = energy_predictions
        teacher_pseudo_df["followup_protocol"] = followup_predictions
        teacher_pseudo_df["precursor_category"] = precursor_predictions
        student_probabilities = fit_student_transformer(train_df, teacher_pseudo_df)
    if student_probabilities is not None:
        variability_weight = STUDENT_BLEND_WEIGHTS["variability_pattern"]
        variability_probabilities = (
            (1.0 - variability_weight) * variability_probabilities
            + variability_weight * student_probabilities["variability_pattern"]
        )
        variability_predictions = variability_bundle.classes[
            variability_probabilities.argmax(axis=1)
        ]

        followup_weight = STUDENT_BLEND_WEIGHTS["followup_protocol"]
        followup_probabilities = (
            (1.0 - followup_weight) * followup_probabilities
            + followup_weight * student_probabilities["followup_protocol"]
        )
        followup_predictions = structured_bundle.followup_classes[
            followup_probabilities.argmax(axis=1)
        ]

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "transient_class": clean_predictions["transient_class"].values,
            "host_environment": clean_predictions["host_environment"].values,
            "spectral_regime": clean_predictions["spectral_regime"].values,
            "variability_pattern": variability_predictions,
            "distance_bin": distance_predictions,
            "energy_tier": energy_predictions,
            "followup_protocol": followup_predictions,
            "precursor_category": precursor_predictions,
        }
    )

    validate_submission(submission, sample_submission)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved submission to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
