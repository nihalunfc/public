# Synthetic Astronomical Transient Reports

![NLP](https://img.shields.io/badge/Domain-NLP-blue) ![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow) ![Status](https://img.shields.io/badge/Status-Accepted-success)

> **Metric:** Higher score is better  
> **Score range:** 0 to 1

---

## Problem Description

### Task
You are given ~90-word prose field reports describing fictional transient events in an invented astronomical survey. Each report contains structured facts woven into natural language. Your task is to predict, for each test report, the values of **8 categorical label columns**: the event's taxonomic class, its host environment, and six physical / operational attributes.

All domain vocabulary—transient class names, host environment names, spectral regimes, protocol codes, precursor categories—is invented for this challenge. No real-world astronomical object, catalog, or survey is referenced.

---

## Dataset

Under `./dataset/public/`:

| File | Contents |
| :--- | :--- |
| `train.csv` | 7,470 rows. Columns: `id`, `narrative`, and the 8 label columns. Use for training. |
| `test.csv` | 2,520 rows. Columns: `id`, `narrative`. Predict labels for every row. |
| `sample_submission.csv` | Template with the correct `id` column and random valid label values. Same schema as the required submission. |

### The 8 Label Columns
*(Identical names across `train.csv`, `sample_submission.csv`, and the final submission)*

| Column | # Values | Quality |
| :--- | :--- | :--- |
| `transient_class` | 12 | Clean (no noise) |
| `host_environment` | 9 | Clean (no noise) |
| `spectral_regime` | 6 | Clean (no noise) |
| `variability_pattern` | 5 | Derived, 12% noise |
| `distance_bin` | 6 | Derived, 12% noise |
| `energy_tier` | 5 | Derived, 12% noise |
| `followup_protocol` | 6 | Derived, 12% noise |
| `precursor_category` | 6 | Derived, 12% noise |

> **Note:** "Derived" columns are functions of the clean columns plus a structured attribute (redshift or luminosity), with 12% of values randomly replaced per label. The theoretical ceiling on any single derived label is ~88% accuracy. Reference the full valid-value sets and the derivation rules in the taxonomy constants at the top of `generate_data.py` (shipped with the dataset).

---

## Compositional Generalization (The Hard Part)

Of the 108 possible `(transient_class, host_environment)` pairs, **18 pairs never appear in training data**—they are reserved entirely for a subset of the test set. Approximately 360 of the 2,520 test reports come from these held-out pairs. The remaining test reports use pairs that were seen in training.

This holdout is the main difficulty axis of the challenge. A model that memorizes `(class, environment)` joint statistics from training will collapse on the held-out pairs. A model that **factorizes** class and environment—learning each as its own concept and composing them—will generalize.

The auxiliary `TRANSIENT_FAMILIES` (4 parents -> 12 classes) and `HOST_ZONES` (3 parents -> 9 environments) mappings in `generate_data.py` are deliberately shipped so that family-level and zone-level auxiliary supervision is available during training.

---

## Evaluation

Your submission must contain all 8 label columns (format validation depends on it, and training on all 8 gives a stronger multi-task signal), but the final score averages Macro F1 over only the **5 derived labels**: 
`variability_pattern`, `distance_bin`, `energy_tier`, `followup_protocol`, `precursor_category`

The 3 clean labels (`transient_class`, `host_environment`, `spectral_regime`) are deliberately excluded from the scoring average because they appear near-verbatim in the narratives and would trivialize the metric. They are still required in the submission and should still be trained on—they provide the multi-task supervision signal that helps your model learn better representations for the derived labels.

For each of the 5 scored label columns, the grader computes:
* `overall_mf1_<col>` — Macro F1 over the full test set for that column
* `unseen_mf1_<col>` — Macro F1 over the `test_unseen` slice for that column

Aggregated across the 5 scored columns and combined:

```text
overall_mf1  = mean(overall_mf1_<col>)   across 5 scored columns  
unseen_mf1   = mean(unseen_mf1_<col>)    across 5 scored columns  
final_score  = 0.3 * overall_mf1 + 0.7 * unseen_mf1
