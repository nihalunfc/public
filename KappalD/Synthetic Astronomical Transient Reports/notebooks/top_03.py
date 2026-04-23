from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "dataset" / "public"
WORKING_DIR = ROOT_DIR / "working"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
METRICS_PATH = WORKING_DIR / "metrics.json"

CLEAN_TARGET_COLUMNS = [
    "transient_class",
    "host_environment",
    "spectral_regime",
]
SCORED_TARGET_COLUMNS = [
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
    "precursor_category",
]
SUBMISSION_COLUMNS = ["id", *CLEAN_TARGET_COLUMNS, *SCORED_TARGET_COLUMNS]

TEXT_MAX_FEATURES = 20_000
RANDOM_SEED = 42
DISTANCE_REDSHIFT_THRESHOLDS = (
    0.04,
    0.14,
    0.35,
    0.75,
    1.80,
)
FOLLOWUP_LOOKUP_COLUMN_SETS = (
    ("spectral_regime", "variability_pattern", "distance_bin"),
    (
        "transient_class",
        "spectral_regime",
        "variability_pattern",
        "distance_bin",
    ),
    ("spectral_regime", "distance_bin"),
    ("spectral_regime", "variability_pattern"),
)

WHITESPACE_PATTERN = re.compile(r"\s+")
REDSHIFT_PATTERN = re.compile(
    r"\bz\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
LUMINOSITY_PATTERN = re.compile(
    r"\blog\s*L\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def make_onehot_encoder(*, sparse_output: bool) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    try:
        return OneHotEncoder(
            sparse_output=sparse_output,
            **kwargs,
        )
    except TypeError:
        return OneHotEncoder(sparse=sparse_output, **kwargs)


class ConstantClassifier:
    def __init__(self, label: str) -> None:
        self.label = str(label)

    def predict(self, features: Any) -> np.ndarray:
        size = (
            int(features.shape[0])
            if hasattr(features, "shape")
            else len(features)
        )
        return np.full(size, self.label, dtype=object)


@dataclass(frozen=True)
class SparseHybridBundle:
    encoder: Any
    model: Any
    categorical_columns: tuple[str, ...]


@dataclass(frozen=True)
class DenseStructuredBundle:
    encoder: Any
    model: Any
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    numeric_fill_values: dict[str, float]


@dataclass(frozen=True)
class MajorityLookupBundle:
    mappings: tuple[tuple[tuple[str, ...], dict[tuple[str, ...], str]], ...]
    default_label: str


@dataclass(frozen=True)
class SolutionArtifacts:
    text_vectorizer: TfidfVectorizer
    clean_models: dict[str, Any]
    variability_bundle: SparseHybridBundle
    distance_bundle: SparseHybridBundle
    distance_missing_bundle: DenseStructuredBundle
    energy_bundle: DenseStructuredBundle
    followup_bundle: MajorityLookupBundle
    followup_unseen_bundle: DenseStructuredBundle
    precursor_bundle: DenseStructuredBundle
    seen_clean_pairs: frozenset[tuple[str, str]]
    label_domains: dict[str, tuple[str, ...]]


def normalize_narrative(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return text


def extract_numeric_value(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(str(text))
    if match is None:
        return None
    return float(match.group(1))


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    narrative = prepared["narrative"].fillna("").astype(str)
    prepared["normalized_narrative"] = narrative.map(normalize_narrative)
    prepared["redshift"] = narrative.map(
        lambda value: extract_numeric_value(REDSHIFT_PATTERN, value)
    )
    prepared["luminosity"] = narrative.map(
        lambda value: extract_numeric_value(LUMINOSITY_PATTERN, value)
    )
    prepared["has_redshift"] = prepared["redshift"].notna().astype(int)
    prepared["has_luminosity"] = prepared["luminosity"].notna().astype(int)
    return prepared


def predict_distance_from_redshift_value(redshift: float | None) -> str | None:
    if redshift is None or pd.isna(redshift):
        return None
    if redshift <= DISTANCE_REDSHIFT_THRESHOLDS[0]:
        return "near"
    if redshift <= DISTANCE_REDSHIFT_THRESHOLDS[1]:
        return "mid_near"
    if redshift <= DISTANCE_REDSHIFT_THRESHOLDS[2]:
        return "mid"
    if redshift <= DISTANCE_REDSHIFT_THRESHOLDS[3]:
        return "mid_far"
    if redshift <= DISTANCE_REDSHIFT_THRESHOLDS[4]:
        return "far"
    return "very_far"


def ensure_dataset_files_exist() -> None:
    required_paths = [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing dataset files: {missing_text}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset_files_exist()
    train_frame = pd.read_csv(TRAIN_PATH)
    test_frame = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    return train_frame, test_frame, sample_submission


def build_text_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )


def fit_sparse_linear_classifier(
    features: Any,
    target: pd.Series,
    *,
    balanced: bool,
    c_value: float,
) -> Any:
    labels = target.astype(str)
    unique_labels = labels.unique()
    if len(unique_labels) == 1:
        return ConstantClassifier(unique_labels[0])
    class_weight = "balanced" if balanced else None
    model = LinearSVC(
        C=c_value,
        class_weight=class_weight,
        dual="auto",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    model.fit(features, labels)
    return model


def fit_dense_logistic_classifier(
    features: np.ndarray,
    target: pd.Series,
) -> Any:
    labels = target.astype(str)
    unique_labels = labels.unique()
    if len(unique_labels) == 1:
        return ConstantClassifier(unique_labels[0])
    model = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    model.fit(features, labels)
    return model


def fit_dense_random_forest_classifier(
    features: np.ndarray,
    target: pd.Series,
) -> Any:
    labels = target.astype(str)
    unique_labels = labels.unique()
    if len(unique_labels) == 1:
        return ConstantClassifier(unique_labels[0])
    model = RandomForestClassifier(
        n_estimators=300,
        n_jobs=1,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )
    model.fit(features, labels)
    return model


def predict_classifier(model: Any, features: Any) -> np.ndarray:
    return np.asarray(model.predict(features), dtype=object)


def fit_sparse_hybrid_bundle(
    text_features: Any,
    categorical_frame: pd.DataFrame,
    target: pd.Series,
    *,
    balanced: bool,
    c_value: float,
) -> SparseHybridBundle:
    encoder = make_onehot_encoder(sparse_output=True)
    encoded_categoricals = encoder.fit_transform(categorical_frame.astype(str))
    features = hstack([text_features, encoded_categoricals], format="csr")
    model = fit_sparse_linear_classifier(
        features,
        target,
        balanced=balanced,
        c_value=c_value,
    )
    return SparseHybridBundle(
        encoder=encoder,
        model=model,
        categorical_columns=tuple(categorical_frame.columns),
    )


def predict_sparse_hybrid_bundle(
    bundle: SparseHybridBundle,
    text_features: Any,
    categorical_frame: pd.DataFrame,
) -> np.ndarray:
    encoded_categoricals = bundle.encoder.transform(
        categorical_frame[list(bundle.categorical_columns)].astype(str)
    )
    features = hstack([text_features, encoded_categoricals], format="csr")
    return predict_classifier(bundle.model, features)


def fit_dense_structured_bundle(
    categorical_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
    target: pd.Series,
    *,
    model_kind: str = "logistic",
) -> DenseStructuredBundle:
    encoder = make_onehot_encoder(sparse_output=False)
    encoded_categoricals = encoder.fit_transform(categorical_frame.astype(str))
    numeric_fill_values = {
        column_name: (
            float(numeric_frame[column_name].median())
            if numeric_frame[column_name].notna().any()
            else 0.0
        )
        for column_name in numeric_frame.columns
    }
    filled_numeric = numeric_frame.fillna(numeric_fill_values).to_numpy(
        dtype=float
    )
    features = np.hstack([encoded_categoricals, filled_numeric])
    if model_kind == "logistic":
        model = fit_dense_logistic_classifier(features, target)
    elif model_kind == "random_forest":
        model = fit_dense_random_forest_classifier(features, target)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    return DenseStructuredBundle(
        encoder=encoder,
        model=model,
        categorical_columns=tuple(categorical_frame.columns),
        numeric_columns=tuple(numeric_frame.columns),
        numeric_fill_values=numeric_fill_values,
    )


def transform_dense_structured_bundle(
    bundle: DenseStructuredBundle,
    categorical_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
) -> np.ndarray:
    encoded_categoricals = bundle.encoder.transform(
        categorical_frame[list(bundle.categorical_columns)].astype(str)
    )
    filled_numeric = numeric_frame[list(bundle.numeric_columns)].fillna(
        bundle.numeric_fill_values
    )
    numeric_values = filled_numeric.to_numpy(dtype=float)
    return np.hstack([encoded_categoricals, numeric_values])


def predict_dense_structured_bundle(
    bundle: DenseStructuredBundle,
    categorical_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
) -> np.ndarray:
    features = transform_dense_structured_bundle(
        bundle,
        categorical_frame,
        numeric_frame,
    )
    return predict_classifier(bundle.model, features)


def build_majority_lookup_mapping(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    target_column: str,
) -> dict[tuple[str, ...], str]:
    mapping: dict[tuple[str, ...], str] = {}
    grouped = frame.groupby(list(key_columns), dropna=False)
    for key, part in grouped:
        normalized_key = key if isinstance(key, tuple) else (key,)
        normalized_key = tuple(str(value) for value in normalized_key)
        mapping[normalized_key] = (
            part[target_column].astype(str).value_counts().index[0]
        )
    return mapping


def fit_majority_lookup_bundle(
    frame: pd.DataFrame,
    *,
    key_column_sets: tuple[tuple[str, ...], ...],
    target_column: str,
) -> MajorityLookupBundle:
    mappings = tuple(
        (
            key_columns,
            build_majority_lookup_mapping(
                frame,
                key_columns=key_columns,
                target_column=target_column,
            ),
        )
        for key_columns in key_column_sets
    )
    default_label = frame[target_column].astype(str).mode()[0]
    return MajorityLookupBundle(
        mappings=mappings,
        default_label=default_label,
    )


def predict_majority_lookup_bundle(
    bundle: MajorityLookupBundle,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    predictions: list[str] = []
    for _, row in feature_frame.iterrows():
        predicted_label = bundle.default_label
        for key_columns, mapping in bundle.mappings:
            key = tuple(str(row[column_name]) for column_name in key_columns)
            if key in mapping:
                predicted_label = mapping[key]
                break
        predictions.append(predicted_label)
    return np.asarray(predictions, dtype=object)


def fit_solution_artifacts(train_frame: pd.DataFrame) -> SolutionArtifacts:
    prepared_train = prepare_frame(train_frame)

    text_vectorizer = build_text_vectorizer(TEXT_MAX_FEATURES)
    text_features = text_vectorizer.fit_transform(
        prepared_train["normalized_narrative"]
    )

    clean_models: dict[str, Any] = {}
    for column_name in CLEAN_TARGET_COLUMNS:
        clean_models[column_name] = fit_sparse_linear_classifier(
            text_features,
            prepared_train[column_name],
            balanced=False,
            c_value=0.75,
        )

    clean_categorical_frame = prepared_train[CLEAN_TARGET_COLUMNS]
    variability_bundle = fit_sparse_hybrid_bundle(
        text_features=text_features,
        categorical_frame=clean_categorical_frame,
        target=prepared_train["variability_pattern"],
        balanced=True,
        c_value=0.75,
    )
    distance_bundle = fit_sparse_hybrid_bundle(
        text_features=text_features,
        categorical_frame=clean_categorical_frame,
        target=prepared_train["distance_bin"],
        balanced=True,
        c_value=0.75,
    )
    distance_missing_categorical_frame = pd.concat(
        [clean_categorical_frame, prepared_train[["variability_pattern"]]],
        axis=1,
    )
    missing_redshift_mask = prepared_train["has_redshift"] == 0
    distance_missing_bundle = fit_dense_structured_bundle(
        categorical_frame=distance_missing_categorical_frame.loc[
            missing_redshift_mask
        ],
        numeric_frame=prepared_train.loc[missing_redshift_mask, []],
        target=prepared_train.loc[missing_redshift_mask, "distance_bin"],
    )

    numeric_columns = [
        "redshift",
        "luminosity",
        "has_redshift",
        "has_luminosity",
    ]
    energy_bundle = fit_dense_structured_bundle(
        categorical_frame=clean_categorical_frame,
        numeric_frame=prepared_train[numeric_columns],
        target=prepared_train["energy_tier"],
        model_kind="random_forest",
    )

    followup_bundle = fit_majority_lookup_bundle(
        prepared_train,
        key_column_sets=FOLLOWUP_LOOKUP_COLUMN_SETS,
        target_column="followup_protocol",
    )
    followup_unseen_bundle = fit_dense_structured_bundle(
        categorical_frame=prepared_train[
            [
                *CLEAN_TARGET_COLUMNS,
                "variability_pattern",
                "distance_bin",
                "energy_tier",
            ]
        ],
        numeric_frame=prepared_train[numeric_columns],
        target=prepared_train["followup_protocol"],
        model_kind="random_forest",
    )
    precursor_bundle = fit_dense_structured_bundle(
        categorical_frame=clean_categorical_frame,
        numeric_frame=prepared_train[[]],
        target=prepared_train["precursor_category"],
    )
    seen_clean_pairs = frozenset(
        zip(
            prepared_train["transient_class"].astype(str),
            prepared_train["host_environment"].astype(str),
        )
    )

    label_domains = {
        column_name: tuple(
            sorted(train_frame[column_name].astype(str).unique().tolist())
        )
        for column_name in SUBMISSION_COLUMNS[1:]
    }

    return SolutionArtifacts(
        text_vectorizer=text_vectorizer,
        clean_models=clean_models,
        variability_bundle=variability_bundle,
        distance_bundle=distance_bundle,
        distance_missing_bundle=distance_missing_bundle,
        energy_bundle=energy_bundle,
        followup_bundle=followup_bundle,
        followup_unseen_bundle=followup_unseen_bundle,
        precursor_bundle=precursor_bundle,
        seen_clean_pairs=seen_clean_pairs,
        label_domains=label_domains,
    )


def predict_clean_targets(
    artifacts: SolutionArtifacts,
    prepared_frame: pd.DataFrame,
    text_features: Any,
) -> pd.DataFrame:
    predictions: dict[str, np.ndarray] = {}
    for column_name in CLEAN_TARGET_COLUMNS:
        predictions[column_name] = predict_classifier(
            artifacts.clean_models[column_name],
            text_features,
        )
    return pd.DataFrame(predictions, index=prepared_frame.index)


def predict_distance_targets(
    artifacts: SolutionArtifacts,
    prepared_frame: pd.DataFrame,
    clean_predictions: pd.DataFrame,
    text_features: Any,
    variability_predictions: np.ndarray,
) -> np.ndarray:
    distance_predictions = predict_sparse_hybrid_bundle(
        artifacts.distance_bundle,
        text_features=text_features,
        categorical_frame=clean_predictions,
    )
    distance_missing_frame = clean_predictions.copy()
    distance_missing_frame["variability_pattern"] = variability_predictions
    missing_distance_predictions = predict_dense_structured_bundle(
        artifacts.distance_missing_bundle,
        categorical_frame=distance_missing_frame,
        numeric_frame=prepared_frame[[]],
    )
    missing_redshift_mask = prepared_frame["redshift"].isna().to_numpy()
    distance_predictions = np.where(
        missing_redshift_mask,
        missing_distance_predictions,
        distance_predictions,
    )

    distance_redshift_predictions = prepared_frame["redshift"].map(
        predict_distance_from_redshift_value
    )
    visible_redshift_mask = distance_redshift_predictions.notna().to_numpy()
    return np.where(
        visible_redshift_mask,
        distance_redshift_predictions.to_numpy(dtype=object),
        distance_predictions,
    )


def predict_followup_targets(
    artifacts: SolutionArtifacts,
    clean_predictions: pd.DataFrame,
    followup_inputs: pd.DataFrame,
    numeric_frame: pd.DataFrame,
) -> np.ndarray:
    lookup_predictions = predict_majority_lookup_bundle(
        artifacts.followup_bundle,
        followup_inputs,
    )
    unseen_predictions = predict_dense_structured_bundle(
        artifacts.followup_unseen_bundle,
        categorical_frame=followup_inputs,
        numeric_frame=numeric_frame,
    )
    predicted_pairs = list(
        zip(
            clean_predictions["transient_class"].astype(str),
            clean_predictions["host_environment"].astype(str),
        )
    )
    unseen_pair_mask = np.asarray(
        [
            predicted_pair not in artifacts.seen_clean_pairs
            for predicted_pair in predicted_pairs
        ],
        dtype=bool,
    )
    return np.where(
        unseen_pair_mask,
        unseen_predictions,
        lookup_predictions,
    )


def predict_scored_targets(
    artifacts: SolutionArtifacts,
    prepared_frame: pd.DataFrame,
    clean_predictions: pd.DataFrame,
    text_features: Any,
) -> pd.DataFrame:
    variability_predictions = predict_sparse_hybrid_bundle(
        artifacts.variability_bundle,
        text_features=text_features,
        categorical_frame=clean_predictions,
    )
    distance_predictions = predict_distance_targets(
        artifacts,
        prepared_frame=prepared_frame,
        clean_predictions=clean_predictions,
        text_features=text_features,
        variability_predictions=variability_predictions,
    )

    numeric_columns = [
        "redshift",
        "luminosity",
        "has_redshift",
        "has_luminosity",
    ]
    numeric_frame = prepared_frame[numeric_columns]

    energy_predictions = predict_dense_structured_bundle(
        artifacts.energy_bundle,
        categorical_frame=clean_predictions,
        numeric_frame=numeric_frame,
    )
    followup_inputs = clean_predictions.copy()
    followup_inputs["variability_pattern"] = variability_predictions
    followup_inputs["distance_bin"] = distance_predictions
    followup_inputs["energy_tier"] = energy_predictions
    followup_predictions = predict_followup_targets(
        artifacts,
        clean_predictions=clean_predictions,
        followup_inputs=followup_inputs,
        numeric_frame=numeric_frame,
    )
    precursor_predictions = predict_dense_structured_bundle(
        artifacts.precursor_bundle,
        categorical_frame=clean_predictions,
        numeric_frame=prepared_frame[[]],
    )

    return pd.DataFrame(
        {
            "variability_pattern": variability_predictions,
            "distance_bin": distance_predictions,
            "energy_tier": energy_predictions,
            "followup_protocol": followup_predictions,
            "precursor_category": precursor_predictions,
        },
        index=prepared_frame.index,
    )


def predict_submission_rows(
    artifacts: SolutionArtifacts,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    prepared_test = prepare_frame(test_frame)
    text_features = artifacts.text_vectorizer.transform(
        prepared_test["normalized_narrative"]
    )
    clean_predictions = predict_clean_targets(
        artifacts,
        prepared_frame=prepared_test,
        text_features=text_features,
    )
    scored_predictions = predict_scored_targets(
        artifacts,
        prepared_frame=prepared_test,
        clean_predictions=clean_predictions,
        text_features=text_features,
    )
    prediction_frame = pd.concat(
        [test_frame[["id"]], clean_predictions, scored_predictions],
        axis=1,
    )
    return prediction_frame[SUBMISSION_COLUMNS]


def build_submission_frame(
    sample_submission: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    submission = sample_submission[["id"]].merge(
        prediction_frame,
        on="id",
        how="left",
        validate="one_to_one",
    )
    submission = submission[SUBMISSION_COLUMNS]
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    label_domains: dict[str, tuple[str, ...]],
) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError("Submission columns do not match expected schema")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count does not match sample")
    if not submission["id"].equals(sample_submission["id"]):
        raise ValueError("Submission ids must preserve sample order")
    if submission[SUBMISSION_COLUMNS[1:]].isnull().any().any():
        raise ValueError("Submission contains missing predictions")
    for column_name, allowed_values in label_domains.items():
        if not submission[column_name].isin(allowed_values).all():
            raise ValueError(
                f"Submission contains invalid values for {column_name}"
            )


def build_metrics_payload(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prepared_train: pd.DataFrame,
    prepared_test: pd.DataFrame,
    submission: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(submission)),
        "train_numeric_coverage": {
            "redshift_rate": float(prepared_train["has_redshift"].mean()),
            "luminosity_rate": float(prepared_train["has_luminosity"].mean()),
        },
        "test_numeric_coverage": {
            "redshift_rate": float(prepared_test["has_redshift"].mean()),
            "luminosity_rate": float(prepared_test["has_luminosity"].mean()),
        },
        "model_stack": {
            "clean_targets": "char_tfidf + LinearSVC",
            "variability_pattern": "char_tfidf + clean_labels + LinearSVC",
            "distance_bin": (
                "char_tfidf + clean_labels + LinearSVC, variability-aware logistic fallback without visible redshift, and redshift-threshold override"
            ),
            "energy_tier": (
                "clean_labels + numeric signals + RandomForestClassifier"
            ),
            "followup_protocol": (
                "majority lookup over spectral + variability + distance with"
                " RandomForest override for unseen predicted clean pairs,"
                " using clean, scored, and numeric signals"
            ),
            "precursor_category": "clean_labels + LogisticRegression",
        },
    }


def run_training_pipeline() -> dict[str, Any]:
    train_frame, test_frame, sample_submission = load_data()
    artifacts = fit_solution_artifacts(train_frame)
    prediction_frame = predict_submission_rows(artifacts, test_frame)
    submission = build_submission_frame(sample_submission, prediction_frame)

    validate_submission(
        submission,
        sample_submission,
        label_domains=artifacts.label_domains,
    )

    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    prepared_train = prepare_frame(train_frame)
    prepared_test = prepare_frame(test_frame)
    metrics_payload = build_metrics_payload(
        train_frame=train_frame,
        test_frame=test_frame,
        prepared_train=prepared_train,
        prepared_test=prepared_test,
        submission=submission,
    )
    METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    return metrics_payload


def main() -> None:
    run_training_pipeline()


if __name__ == "__main__":
    main()
