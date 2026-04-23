from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset" / "public"
WORKING_DIR = ROOT / "working"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMNS = [
    "transient_class",
    "host_environment",
    "spectral_regime",
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
    "precursor_category",
]

CLASS_VALUES = [
    "Cryogenic Afterglow",
    "Dark Reverberator",
    "Helicity Collapse",
    "Hot Jet Eruption",
    "Hyperaccretion Flare",
    "Limb-Cycle Pulsar",
    "Lithogen Burst",
    "Neutronfall",
    "Quasi-Periodic Echoer",
    "Spectral Ghost",
    "Tidal Spectacle",
    "Tidally Locked Beacon",
]

ENVIRONMENT_PATTERNS = {
    "AGN Wind Region": [r"\bagn wind region\b", r"\bagn wind\b"],
    "Circumnuclear Disk": [r"\bcircumnuclear disk\b"],
    "Diffuse Warm Medium": [r"\bdiffuse warm medium\b", r"\bdwm\b"],
    "Galactic Bar Vicinity": [r"\bgalactic bar vicinity\b"],
    "Globular Cluster Core": [
        r"\bglobular cluster core\b",
        r"\bcore of a globular cluster\b",
        r"\bcore region of a globular cluster\b",
        r"\bwithin the core of a globular cluster\b",
    ],
    "Intergalactic Halo": [r"\bintergalactic halo\b"],
    "Nuclear Star Cluster": [r"\bnuclear star cluster\b"],
    "Stellar Stream": [r"\bstellar stream\b", r"\bstellar stream region\b"],
    "Young Stellar Association": [r"\byoung stellar association\b"],
}

SPECTRAL_PATTERNS = {
    "hard_xray": [r"\bhard x-ray\b", r"\bhard xray\b", r"\bhard_xray\b"],
    "infrared": [r"\binfrared\b"],
    "optical": [r"\boptical\b"],
    "radio": [r"\bradio\b"],
    "soft_xray": [r"\bsoft x-ray\b", r"\bsoft xray\b", r"\bsoft_xray\b"],
    "uv": [r"\bultraviolet\b", r"\buv\b"],
}

VARIABILITY_PATTERNS = {
    "quasi_periodic": [r"\bquasi-periodic\b", r"\bquasi periodic\b"],
    "monotonic_rise": [
        r"\bmonotonic rise\b",
        r"\bcontinuous increase in brightness\b",
        r"\bconsistent monotonic rise\b",
        r"\bclear and consistent monotonic rise\b",
    ],
    "double_peak": [r"\bdouble-peaked\b", r"\bdouble peaked\b", r"\bdouble peak\b"],
    "chaotic": [r"\bchaotic\b"],
    "flat": [r"\bflat\b", r"\bstable brightness\b", r"\bconstant brightness\b", r"\blargely constant brightness\b"],
}

Z_PATTERNS = [
    r"\bz\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    r"\bredshift of z\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    r"\bredshift[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)",
]

LOGL_PATTERNS = [
    r"\blog\s*L\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    r"\blog luminosity of\s*([0-9]+(?:\.[0-9]+)?)",
    r"\bluminosity of log\s*L\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    r"\bluminosity[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)",
]

TARGET_WEIGHTS = {
    "variability_pattern": (0.60, 0.40),
    "distance_bin": (0.30, 0.70),
    "energy_tier": (0.30, 0.70),
    "followup_protocol": (0.25, 0.75),
    "precursor_category": (0.20, 0.80),
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def first_match(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_numeric(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_transient_class(text: str) -> str | None:
    lowered = normalize_text(text)
    for value in sorted(CLASS_VALUES, key=len, reverse=True):
        if value.lower() in lowered:
            return value
    return None


def extract_host_environment(text: str) -> str | None:
    lowered = normalize_text(text)
    for label, patterns in ENVIRONMENT_PATTERNS.items():
        if first_match(lowered, patterns):
            return label
    return None


def extract_spectral_regime(text: str) -> str | None:
    lowered = normalize_text(text)
    matches = [label for label, patterns in SPECTRAL_PATTERNS.items() if first_match(lowered, patterns)]
    if len(matches) == 1:
        return matches[0]
    return None


def extract_variability(text: str) -> str | None:
    lowered = normalize_text(text)
    matches = [label for label, patterns in VARIABILITY_PATTERNS.items() if first_match(lowered, patterns)]
    if len(matches) == 1:
        return matches[0]
    return None


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["narrative_norm"] = out["narrative"].fillna("").map(normalize_text)
    out["z_value"] = out["narrative_norm"].map(lambda text: extract_numeric(text, Z_PATTERNS))
    out["log_l_value"] = out["narrative_norm"].map(lambda text: extract_numeric(text, LOGL_PATTERNS))
    out["transient_class_hint"] = out["narrative_norm"].map(extract_transient_class)
    out["host_environment_hint"] = out["narrative_norm"].map(extract_host_environment)
    out["spectral_regime_hint"] = out["narrative_norm"].map(extract_spectral_regime)
    out["variability_hint"] = out["narrative_norm"].map(extract_variability)
    out["text_length"] = out["narrative"].fillna("").str.len()
    return out


def fit_text_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2500, C=4.0, n_jobs=None)),
        ]
    ).fit(X_train, y_train)


def fit_structured_model(train_df: pd.DataFrame, target: str) -> CatBoostClassifier:
    feature_cols = [
        "transient_class",
        "host_environment",
        "spectral_regime",
        "z_value",
        "log_l_value",
        "text_length",
    ]
    if target in {"followup_protocol", "precursor_category"}:
        feature_cols.extend(["variability_pattern", "distance_bin", "energy_tier"])
    cat_features = [0, 1, 2]
    if target in {"followup_protocol", "precursor_category"}:
        cat_features.extend([6, 7, 8])

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=160,
        depth=6,
        learning_rate=0.08,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_df[feature_cols], train_df[target], cat_features=cat_features)
    return model


def fill_with_text_fallback(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    target: str,
    hint_col: str,
) -> np.ndarray:
    text_model = fit_text_model(train_df["narrative_norm"], train_df[target])
    pred = text_model.predict(infer_df["narrative_norm"]).astype(object)
    if hint_col:
        hint_values = infer_df[hint_col]
        mask = hint_values.notna()
        pred[mask.to_numpy()] = hint_values[mask].to_numpy()
    return pred


def align_and_blend(
    text_classes: np.ndarray,
    text_proba: np.ndarray,
    struct_classes: np.ndarray,
    struct_proba: np.ndarray,
    label_order: list[str],
    text_weight: float,
    struct_weight: float,
) -> np.ndarray:
    aligned = np.zeros((text_proba.shape[0], len(label_order)), dtype=float)
    text_index = {label: idx for idx, label in enumerate(text_classes)}
    struct_index = {label: idx for idx, label in enumerate(struct_classes)}
    for idx, label in enumerate(label_order):
        if label in text_index:
            aligned[:, idx] += text_weight * text_proba[:, text_index[label]]
        if label in struct_index:
            aligned[:, idx] += struct_weight * struct_proba[:, struct_index[label]]
    return aligned


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    train = add_engineered_columns(train)
    test = add_engineered_columns(test)

    predicted = pd.DataFrame({"id": test["id"]})

    for label, hint_col in [
        ("transient_class", "transient_class_hint"),
        ("host_environment", "host_environment_hint"),
        ("spectral_regime", "spectral_regime_hint"),
    ]:
        predicted[label] = fill_with_text_fallback(train, test, label, hint_col)

    struct_train_base = train.copy()
    struct_test_base = test.copy()
    struct_test_base["transient_class"] = predicted["transient_class"]
    struct_test_base["host_environment"] = predicted["host_environment"]
    struct_test_base["spectral_regime"] = predicted["spectral_regime"]

    for target in ["variability_pattern", "distance_bin", "energy_tier"]:
        text_model = fit_text_model(train["narrative_norm"], train[target])
        struct_model = fit_structured_model(struct_train_base[[
            "transient_class",
            "host_environment",
            "spectral_regime",
            "z_value",
            "log_l_value",
            "text_length",
            target,
        ]], target)

        text_proba = text_model.predict_proba(test["narrative_norm"])
        struct_proba = struct_model.predict_proba(struct_test_base[[
            "transient_class",
            "host_environment",
            "spectral_regime",
            "z_value",
            "log_l_value",
            "text_length",
        ]])
        label_order = sorted(train[target].unique().tolist())
        text_weight, struct_weight = TARGET_WEIGHTS[target]
        blended = align_and_blend(
            text_model.classes_,
            text_proba,
            struct_model.classes_,
            struct_proba,
            label_order,
            text_weight,
            struct_weight,
        )
        pred = np.array(label_order)[blended.argmax(axis=1)].astype(object)

        if target == "variability_pattern":
            hint_values = test["variability_hint"]
            mask = hint_values.notna()
            pred[mask.to_numpy()] = hint_values[mask].to_numpy()

        predicted[target] = pred
        struct_test_base[target] = pred

    for target in ["followup_protocol", "precursor_category"]:
        text_model = fit_text_model(train["narrative_norm"], train[target])
        struct_model = fit_structured_model(struct_train_base[[
            "transient_class",
            "host_environment",
            "spectral_regime",
            "z_value",
            "log_l_value",
            "text_length",
            "variability_pattern",
            "distance_bin",
            "energy_tier",
            target,
        ]], target)

        text_proba = text_model.predict_proba(test["narrative_norm"])
        struct_proba = struct_model.predict_proba(struct_test_base[[
            "transient_class",
            "host_environment",
            "spectral_regime",
            "z_value",
            "log_l_value",
            "text_length",
            "variability_pattern",
            "distance_bin",
            "energy_tier",
        ]])
        label_order = sorted(train[target].unique().tolist())
        text_weight, struct_weight = TARGET_WEIGHTS[target]
        blended = align_and_blend(
            text_model.classes_,
            text_proba,
            struct_model.classes_,
            struct_proba,
            label_order,
            text_weight,
            struct_weight,
        )
        pred = np.array(label_order)[blended.argmax(axis=1)]
        predicted[target] = pred
        struct_test_base[target] = pred

    submission = predicted[["id", *LABEL_COLUMNS]].copy()
    submission = submission[sample.columns.tolist()]

    allowed_values = {
        column: set(train[column].dropna().unique().tolist())
        for column in LABEL_COLUMNS
    }
    for column in LABEL_COLUMNS:
        invalid_mask = ~submission[column].isin(allowed_values[column])
        if invalid_mask.any():
            fallback_value = train[column].mode().iloc[0]
            submission.loc[invalid_mask, column] = fallback_value

    submission.to_csv(WORKING_DIR / "submission.csv", index=False)

    summary = {
        "submission_path": str(WORKING_DIR / "submission.csv"),
        "rows": int(len(submission)),
        "columns": submission.columns.tolist(),
    }
    (WORKING_DIR / "submission_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
