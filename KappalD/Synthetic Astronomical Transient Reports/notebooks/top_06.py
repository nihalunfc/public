import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


DATA_DIR = "./dataset/public"
WORK_DIR = "./working"

ID_COL = "id"
TEXT_COL = "narrative"

CLEAN_COLS = ["transient_class", "host_environment", "spectral_regime"]
SCORED_COLS = [
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
    "precursor_category",
]
ALL_LABEL_COLS = CLEAN_COLS + SCORED_COLS

SUBMISSION_COLUMNS = [
    "id",
    "transient_class",
    "host_environment",
    "spectral_regime",
    "variability_pattern",
    "distance_bin",
    "energy_tier",
    "followup_protocol",
    "precursor_category",
]

RED_PATTERNS = [
    r"redshift(?: of)?(?: z)?\s*=\s*([0-9]+\.[0-9]+)",
    r"at redshift z\s*=\s*([0-9]+\.[0-9]+)",
    r"redshift z\s*=\s*([0-9]+\.[0-9]+)",
]
LUM_PATTERNS = [r"log L\s*=\s*([0-9]+\.[0-9]+)"]

VARIABILITY_HINTS = {
    "chaotic": ["chaotic"],
    "double_peak": ["double peak", "double-peaked"],
    "flat": ["flat", "constant"],
    "monotonic_rise": ["monotonic rise", "monotonic increase", "continuous increase"],
    "quasi_periodic": ["quasi-periodic", "quasi periodic"],
}


def build_alias_maps():
    transient_aliases = {
        "cryogenic afterglow": "Cryogenic Afterglow",
        "dark reverberator": "Dark Reverberator",
        "helicity collapse": "Helicity Collapse",
        "hot jet eruption": "Hot Jet Eruption",
        "hyperaccretion flare": "Hyperaccretion Flare",
        "limb-cycle pulsar": "Limb-Cycle Pulsar",
        "lithogen burst": "Lithogen Burst",
        "neutronfall": "Neutronfall",
        "quasi-periodic echoer": "Quasi-Periodic Echoer",
        "qpe": "Quasi-Periodic Echoer",
        "spectral ghost": "Spectral Ghost",
        "tidal spectacle": "Tidal Spectacle",
        "tidally locked beacon": "Tidally Locked Beacon",
    }
    host_aliases = {
        "agn wind region": "AGN Wind Region",
        "circumnuclear disk": "Circumnuclear Disk",
        "diffuse warm medium": "Diffuse Warm Medium",
        "dwm": "Diffuse Warm Medium",
        "galactic bar vicinity": "Galactic Bar Vicinity",
        "globular cluster core": "Globular Cluster Core",
        "intergalactic halo": "Intergalactic Halo",
        "nuclear star cluster": "Nuclear Star Cluster",
        "stellar stream": "Stellar Stream",
        "young stellar association": "Young Stellar Association",
    }
    spectral_aliases = {
        "hard x-ray": "hard_xray",
        "hard x-rays": "hard_xray",
        "infrared": "infrared",
        "optical": "optical",
        "radio": "radio",
        "soft x-ray": "soft_xray",
        "soft x-rays": "soft_xray",
        "ultraviolet (uv)": "uv",
        "ultraviolet": "uv",
        "uv": "uv",
    }
    return {
        "transient_class": transient_aliases,
        "host_environment": host_aliases,
        "spectral_regime": spectral_aliases,
    }


ALIAS_MAPS = build_alias_maps()


def extract_float(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return np.nan


def extract_variability_hint(text):
    lowered = text.lower()
    for label, aliases in VARIABILITY_HINTS.items():
        if any(alias in lowered for alias in aliases):
            return label
    return "UNKNOWN"


def exact_extract_label(text, alias_to_label):
    lowered = text.lower()
    matches = []
    for alias, label in alias_to_label.items():
        if alias in lowered:
            matches.append((len(alias), label))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def add_engineered_features(df):
    out = df.copy()
    out["redshift"] = out[TEXT_COL].map(lambda x: extract_float(x, RED_PATTERNS))
    out["luminosity"] = out[TEXT_COL].map(lambda x: extract_float(x, LUM_PATTERNS))
    out["redshift_missing"] = out["redshift"].isna().astype(int)
    out["luminosity_missing"] = out["luminosity"].isna().astype(int)
    out["var_hint"] = out[TEXT_COL].map(extract_variability_hint)
    return out


def fit_clean_text_models(train_df):
    models = {}
    for col in CLEAN_COLS:
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 6),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=3000, C=6.0)),
            ]
        )
        model.fit(train_df[TEXT_COL], train_df[col])
        models[col] = model
    return models


def predict_clean_labels(train_df, test_df):
    models = fit_clean_text_models(train_df)
    preds = {}
    for col in CLEAN_COLS:
        model_preds = models[col].predict(test_df[TEXT_COL])
        exact_preds = test_df[TEXT_COL].map(lambda x: exact_extract_label(x, ALIAS_MAPS[col]))
        final_preds = pd.Series(model_preds, index=test_df.index, dtype=object)
        final_preds.loc[exact_preds.notna()] = exact_preds.loc[exact_preds.notna()]
        preds[col] = final_preds.astype(str)
    return preds


def train_and_predict_catboost(train_feat, train_target, test_feat, cat_cols):
    from catboost import CatBoostClassifier

    cat_idx = [train_feat.columns.get_loc(col) for col in cat_cols]
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    model.fit(train_feat, train_target, cat_features=cat_idx)
    preds = model.predict(test_feat).reshape(-1)
    return preds


def train_and_predict_lightgbm(train_feat, train_target, test_feat, cat_cols):
    from lightgbm import LGBMClassifier

    train_feat = train_feat.copy()
    test_feat = test_feat.copy()

    for col in cat_cols:
        categories = pd.Categorical(
            pd.concat([train_feat[col].astype(str), test_feat[col].astype(str)], axis=0).unique()
        ).categories
        train_feat[col] = pd.Categorical(train_feat[col].astype(str), categories=categories)
        test_feat[col] = pd.Categorical(test_feat[col].astype(str), categories=categories)

    y_cat = pd.Categorical(train_target)
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    cat_idx = [train_feat.columns.get_loc(col) for col in cat_cols]
    model.fit(train_feat, y_cat.codes, categorical_feature=cat_idx)
    pred_codes = model.predict(test_feat)
    return y_cat.categories[pred_codes]


def predict_scored_labels(train_df, test_df, clean_test_preds):
    train_feat = train_df[["redshift", "luminosity", "redshift_missing", "luminosity_missing", "var_hint"]].copy()
    test_feat = test_df[["redshift", "luminosity", "redshift_missing", "luminosity_missing", "var_hint"]].copy()

    for col in CLEAN_COLS:
        train_feat[col] = train_df[col].astype(str)
        test_feat[col] = clean_test_preds[col].astype(str)

    cat_cols = ["var_hint"] + CLEAN_COLS

    preds = {}
    use_catboost = True
    try:
        import catboost  # noqa: F401
    except Exception:
        use_catboost = False

    for col in SCORED_COLS:
        if use_catboost:
            preds[col] = train_and_predict_catboost(train_feat, train_df[col], test_feat, cat_cols)
        else:
            preds[col] = train_and_predict_lightgbm(train_feat, train_df[col], test_feat, cat_cols)
    return preds


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train = add_engineered_features(train)
    test = add_engineered_features(test)

    clean_test_preds = predict_clean_labels(train, test)
    scored_test_preds = predict_scored_labels(train, test, clean_test_preds)

    submission = pd.DataFrame({ID_COL: test[ID_COL]})
    for col in CLEAN_COLS:
        submission[col] = clean_test_preds[col].values
    for col in SCORED_COLS:
        submission[col] = scored_test_preds[col]

    submission = submission[SUBMISSION_COLUMNS]
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
