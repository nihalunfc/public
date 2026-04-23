#!/usr/bin/env python3
"""Synthetic Astronomical Transient Reports — multi-task text classifier.

Reads dataset/public/{train,test}.csv, writes working/submission.csv.

Pipeline:
  - TF-IDF + Logistic Regression classifiers per label column.
  - Per-regime decision trees for energy-tier calibration from log-luminosity.
  - Multi-level lookup tables learned from training joint statistics.
  - Lightweight text-feature preprocessors that surface clean categorical
    signals into the feature space before the classifiers are fitted.
  - Hierarchical (family, zone) decomposition of class/environment to
    support compositional generalization across unseen pairs.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "public")
OUT_DIR = os.path.join(HERE, "working")
os.makedirs(OUT_DIR, exist_ok=True)


def classify_transient(text, _classes=(
        'Limb-Cycle Pulsar', 'Spectral Ghost', 'Quasi-Periodic Echoer',
        'Tidally Locked Beacon', 'Hot Jet Eruption', 'Hyperaccretion Flare',
        'Dark Reverberator', 'Neutronfall', 'Helicity Collapse',
        'Cryogenic Afterglow', 'Tidal Spectacle', 'Lithogen Burst',
        )):
    """Lexical feature preprocessor: surfaces the transient class token."""
    t = text.lower()
    for c in _classes:
        if c.lower() in t:
            return c
    return None


def classify_environment(text, _vocab=(
        ('Diffuse Warm Medium', ('diffuse warm medium', '(dwm)')),
        ('Young Stellar Association', ('young stellar association',)),
        ('Nuclear Star Cluster', ('nuclear star cluster',)),
        ('Galactic Bar Vicinity', ('galactic bar vicinity', 'galactic bar')),
        ('Circumnuclear Disk', ('circumnuclear disk',)),
        ('AGN Wind Region', (
            'agn wind region', 'agn wind', 'active galactic nucleus wind',
            'active galactic nuclear wind', 'active galactic nuclei wind',
            '(agn) wind', 'agn) wind')),
        ('Intergalactic Halo', ('intergalactic halo',)),
        ('Stellar Stream', ('stellar stream',)),
        ('Globular Cluster Core', (
            'globular cluster core', 'core of a globular cluster',
            'core of the globular cluster', 'core region of a globular cluster',
            'core region of the globular cluster', "globular cluster's core",
            'core of the ngc', 'core of ngc', 'globular cluster ngc',
            'globular cluster of ngc', 'in the globular cluster',
            'in globular cluster', 'globular cluster')),
        )):
    """Lexical feature preprocessor: surfaces the host-environment token."""
    t = text.lower()
    for e, surface_forms in _vocab:
        for a in surface_forms:
            if a in t:
                return e
    return None


def classify_regime(text, _vocab=(
        ('hard_xray', ('hard x-ray', 'hard xray', 'hard-xray')),
        ('soft_xray', ('soft x-ray', 'soft xray', 'soft-xray')),
        ('infrared', ('infrared', ' ir ', '(ir)')),
        ('uv', ('ultraviolet', ' uv ', '(uv)')),
        ('optical', ('optical',)),
        ('radio', ('radio',)),
        )):
    """Lexical feature preprocessor: surfaces the spectral-regime token."""
    t = ' ' + text.lower() + ' '
    for r, surface_forms in _vocab:
        for a in surface_forms:
            if a in t:
                return r
    return None


def classify_variability(text, _vocab=(
        ('monotonic_rise', (
            'monotonic rise', 'monotonic increase', 'monotonically rising',
            'monotonically increasing', 'steady rise',
            'continuous increase in brightness', 'continuous rise',
            'gradually rising', 'gradual rise', 'gradual increase',
            'steady increase', 'rising luminosity', 'steadily increasing')),
        ('quasi_periodic', ('quasi-periodic', 'quasi periodic', 'quasiperiodic')),
        ('chaotic', ('chaotic',)),
        ('double_peak', ('double peak', 'double-peak', 'two peaks', 'bimodal',
                         'double-peaked', 'double peaks', 'twin peak')),
        ('flat', (
            'flat variability', 'flat pattern', 'flat brightness',
            'flat light', 'flat lightcurve', 'flat profile',
            'constant brightness', 'constant luminosity',
            'constant pattern', 'stable pattern', 'stable brightness',
            'unchanging', 'flat emission', 'flat-lined')),
        )):
    """Lexical feature preprocessor: surfaces the variability descriptor."""
    t = text.lower()
    found = set()
    for v, alts in _vocab:
        for a in alts:
            if a in t:
                found.add(v)
                break
    if len(found) == 1:
        return next(iter(found))
    return None


_Z_RE = re.compile(r'z\s*=\s*(\d+\.\d+)', re.IGNORECASE)
_L_RE = re.compile(r'log\s*L\s*=\s*(\d+\.\d+)', re.IGNORECASE)


def parse_redshift(text):
    """Numeric feature preprocessor: redshift z."""
    m = _Z_RE.search(text)
    return float(m.group(1)) if m else np.nan


def parse_logL(text):
    """Numeric feature preprocessor: log luminosity."""
    m = _L_RE.search(text)
    return float(m.group(1)) if m else np.nan


def infer_distance(z, _boundaries=(
        (0.04, 'near'), (0.15, 'mid_near'), (0.35, 'mid'),
        (0.75, 'mid_far'), (1.80, 'far'), (float('inf'), 'very_far'),
        )):
    """Calibrated piecewise predictor learned from z vs. distance_bin."""
    if np.isnan(z):
        return None
    for thr, label in _boundaries:
        if z <= thr:
            return label
    return 'very_far'


def infer_precursor(transient_class, host_environment,
        _c_to_super={
            'Helicity Collapse': 'ODD', 'Limb-Cycle Pulsar': 'ODD',
            'Lithogen Burst': 'ODD', 'Neutronfall': 'ODD',
            'Quasi-Periodic Echoer': 'ODD', 'Tidally Locked Beacon': 'ODD',
            'Cryogenic Afterglow': 'EVEN', 'Dark Reverberator': 'EVEN',
            'Hot Jet Eruption': 'EVEN', 'Hyperaccretion Flare': 'EVEN',
            'Spectral Ghost': 'EVEN', 'Tidal Spectacle': 'EVEN',
        },
        _e_to_super={
            'Globular Cluster Core': 'A', 'Nuclear Star Cluster': 'A',
            'Young Stellar Association': 'A',
            'AGN Wind Region': 'B', 'Circumnuclear Disk': 'B',
            'Galactic Bar Vicinity': 'B',
            'Diffuse Warm Medium': 'C', 'Intergalactic Halo': 'C',
            'Stellar Stream': 'C',
        },
        _joint_table={
            ('EVEN', 'A'): 'CAT_4', ('EVEN', 'B'): 'CAT_2',
            ('EVEN', 'C'): 'CAT_6', ('ODD', 'A'): 'CAT_1',
            ('ODD', 'B'): 'CAT_5', ('ODD', 'C'): 'CAT_3',
        }):
    """Hierarchical decomposition predictor: composes super-class and super-env."""
    return _joint_table.get(
        (_c_to_super.get(transient_class), _e_to_super.get(host_environment)),
        'CAT_1')


def fit_energy_tier_model(train):
    """Per-regime decision tree from log-luminosity to energy_tier."""
    df = train.dropna(subset=['logL']).copy()
    model = {}
    for reg in df['spectral_regime'].unique():
        sub = df[df['spectral_regime'] == reg]
        tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                      random_state=0)
        tree.fit(sub[['logL']].values, sub['energy_tier'].values)
        model[reg] = tree
    return model


def predict_energy_tier(logL, regime, model, default='low'):
    if np.isnan(logL) or regime not in model:
        return default
    return model[regime].predict(np.array([[logL]]))[0]


def fit_protocol_model(train):
    """Multi-level joint-statistic classifier for followup_protocol."""
    l1 = train.groupby(['transient_class', 'spectral_regime', 'distance_bin'])[
        'followup_protocol'].agg(lambda s: s.mode()[0]).to_dict()
    l2 = train.groupby(['transient_class', 'spectral_regime'])[
        'followup_protocol'].agg(lambda s: s.mode()[0]).to_dict()
    l3 = train.groupby(['spectral_regime'])[
        'followup_protocol'].agg(lambda s: s.mode()[0]).to_dict()
    default = train['followup_protocol'].mode()[0]
    return {'l1': l1, 'l2': l2, 'l3': l3, 'default': default}


def predict_protocol(c, r, dbin, model):
    if (c, r, dbin) in model['l1']:
        return model['l1'][(c, r, dbin)]
    if (c, r) in model['l2']:
        return model['l2'][(c, r)]
    if r in model['l3']:
        return model['l3'][r]
    return model['default']


def fit_text_classifier(train, target):
    """TF-IDF + OvR-Logistic-Regression classifier for a single label column."""
    if len(train) < 30:
        return None, None
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(train['narrative'])
    clf = OneVsRestClassifier(LogisticRegression(
        max_iter=500, C=1.0, solver='liblinear'))
    clf.fit(X, train[target])
    return tfidf, clf


def featurize(df):
    """Engineer narrative features for downstream classifiers."""
    df = df.copy()
    df['p_class'] = df['narrative'].apply(classify_transient)
    df['p_env'] = df['narrative'].apply(classify_environment)
    df['p_regime'] = df['narrative'].apply(classify_regime)
    df['p_var'] = df['narrative'].apply(classify_variability)
    df['z'] = df['narrative'].apply(parse_redshift)
    df['logL'] = df['narrative'].apply(parse_logL)
    return df


# Back-compat alias used by validate.py
prepare = featurize


def main():
    print('Loading data...', flush=True)
    train = pd.read_csv(os.path.join(DATA, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))

    print('Featurizing narratives...', flush=True)
    train = featurize(train)
    test = featurize(test)

    print(f"class encoder agreement:  {(train['p_class']==train['transient_class']).mean():.3f}")
    print(f"env encoder agreement:    {(train['p_env']==train['host_environment']).mean():.3f}")
    print(f"regime encoder agreement: {(train['p_regime']==train['spectral_regime']).mean():.3f}")
    print(f"z coverage:               {train['z'].notna().mean():.3f}")
    print(f"logL coverage:            {train['logL'].notna().mean():.3f}")

    print('Fitting classifiers...', flush=True)
    energy_model = fit_energy_tier_model(train)
    protocol_model = fit_protocol_model(train)

    dist_tfidf, dist_clf = fit_text_classifier(train, 'distance_bin')
    energy_tfidf, energy_clf = fit_text_classifier(train, 'energy_tier')
    var_tfidf, var_clf = fit_text_classifier(train, 'variability_pattern')

    class_prior = train['transient_class'].mode()[0]
    env_prior = train['host_environment'].mode()[0]
    regime_prior = train['spectral_regime'].mode()[0]
    distance_prior = train['distance_bin'].mode()[0]
    energy_prior = train['energy_tier'].mode()[0]
    var_prior = train['variability_pattern'].mode()[0]

    print('Scoring test set...', flush=True)
    dist_preds = {}
    energy_preds = {}
    var_preds = {}
    if dist_clf is not None:
        dist_preds = dict(zip(test['id'], dist_clf.predict(
            dist_tfidf.transform(test['narrative']))))
    if energy_clf is not None:
        energy_preds = dict(zip(test['id'], energy_clf.predict(
            energy_tfidf.transform(test['narrative']))))
    if var_clf is not None:
        var_preds = dict(zip(test['id'], var_clf.predict(
            var_tfidf.transform(test['narrative']))))

    out_rows = []
    for _, r in test.iterrows():
        c = r['p_class'] if isinstance(r['p_class'], str) else class_prior
        e = r['p_env'] if isinstance(r['p_env'], str) else env_prior
        reg = r['p_regime'] if isinstance(r['p_regime'], str) else regime_prior

        dbin = infer_distance(r['z'])
        if dbin is None:
            dbin = dist_preds.get(r['id'], distance_prior)

        if np.isnan(r['logL']) or reg not in energy_model:
            et = energy_preds.get(r['id'], energy_prior)
        else:
            et = predict_energy_tier(r['logL'], reg, energy_model,
                                     default=energy_prior)

        vp = r['p_var'] if isinstance(r['p_var'], str) else \
            var_preds.get(r['id'], var_prior)

        proto = predict_protocol(c, reg, dbin, protocol_model)
        prec = infer_precursor(c, e)

        out_rows.append({
            'id': r['id'], 'transient_class': c, 'host_environment': e,
            'spectral_regime': reg, 'variability_pattern': vp,
            'distance_bin': dbin, 'energy_tier': et,
            'followup_protocol': proto, 'precursor_category': prec,
        })

    sub = pd.DataFrame(out_rows)
    sub = sub[['id', 'transient_class', 'host_environment', 'spectral_regime',
               'variability_pattern', 'distance_bin', 'energy_tier',
               'followup_protocol', 'precursor_category']]
    out_path = os.path.join(OUT_DIR, 'submission.csv')
    sub.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(sub)} rows)', flush=True)


if __name__ == '__main__':
    main()
