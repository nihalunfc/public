# Winning Strategies: Synthetic Astronomical Transient Reports

![Strategy](https://img.shields.io/badge/Focus-Compositional_Generalization-purple) ![Target](https://img.shields.io/badge/Target_Score-0.70+-success)

To achieve a high score in this challenge, you must tackle the author's primary hurdle: **Compositional Generalization**. Because the final metric heavily weights unseen `(class, environment)` pairs (70% of the score), standard models that simply memorize training data distribution will fail. 

Below are the most effective approaches to solve this task and get closer to the ~0.82 theoretical ceiling, along with the reasoning based on the author's rules.

---

## 1. The Validation Strategy (Crucial First Step)

### Approach: Compositional Cross-Validation
**Do not use random splitting or standard K-Fold.** Instead, create a custom validation scheme that mimics the hidden test set. 
1. Identify all unique `(transient_class, host_environment)` pairs in the training data.
2. Hold out a subset of these pairs *entirely* for your validation set. 
3. Ensure no records with these specific pairs exist in your training folds.

**Why this works:** The author evaluates your model heavily (0.7 weight) on 18 held-out pairs. If you use a random split, your local validation score will look amazing (~0.80+) but will collapse on the leaderboard. A compositional split forces your local validation metric to reflect the actual leaderboard environment, allowing you to tune hyperparameters accurately.

---

## 2. Architecture: Multi-Task Learning (MTL)

### Approach: Fine-tuned Transformer with 10 Heads
Use a strong pre-trained encoder like **DeBERTa-v3** or **RoBERTa**. Instead of predicting just the 5 derived labels, attach **10 classification heads** to the final hidden state `[CLS]` token:
* **5 Heads:** The scored derived columns (e.g., `variability_pattern`).
* **3 Heads:** The clean columns (`transient_class`, `host_environment`, `spectral_regime`).
* **2 Auxiliary Heads:** The `TRANSIENT_FAMILIES` and `HOST_ZONES` parents (derived from `generate_data.py`).

**Why this works:**
The author explicitly notes: *"training on all 8 gives a stronger multi-task signal."* The clean labels are present verbatim in the text. By forcing the network to predict the clean labels and their parent hierarchies, you ground the model's textual embeddings in the underlying domain ontology. This rich representation makes it much easier for the derived heads to learn the complex mapping rules, even in the presence of 12% label noise.

---

## 3. Breaking Spurious Correlations (The Core Fix)

### Approach A: Data Augmentation (Entity Swapping)
Since this is a synthetic dataset generated from rules, the text likely follows templated structures. 
1. Extract the explicit mentions of the `transient_class` and `host_environment` in the training narratives.
2. Randomly swap them with other valid classes and environments to artificially generate the "unseen" pairs. 
3. Note: Only use this for training the representations, as swapping them might invalidate the logic for the *derived* labels.

### Approach B: Disentangled Representation Learning
Use techniques like **Dropout**, **Word Dropout**, or **Masked Language Modeling (MLM)** on the specific entity keywords during fine-tuning. Mask out the `host_environment` occasionally so the model is forced to predict the derived labels based on the surrounding context rather than relying exclusively on the joint presence of a specific class and environment.

**Why this works:**
Neural networks are lazy; they will memorize that `Lithogen Burst` + `Globular Cluster Core` = `PROTOCOL_D`. When they see `Lithogen Burst` + `Stellar Stream` (an unseen pair), they guess blindly. Augmentation and masking force the model to factorize the concepts—learning what a "Lithogen Burst" implies on its own, and what a "Stellar Stream" implies on its own.

---

## 4. Handling Label Noise

### Approach: Label Smoothing or Robust Loss Functions
The 5 scored labels have a deliberate 12% noise rate. Standard Cross-Entropy loss will cause the model to overfit to this noise, degrading generalized performance.
* Apply **Label Smoothing** (e.g., `smoothing=0.1`) to prevent the model from becoming overconfident.
* Alternatively, use noise-robust loss functions like **Symmetric Cross Entropy (SCE)** or **Early Stopping** based strictly on your compositional validation set to prevent memorization of noisy artifacts.

**Why this works:**
The author states there is a *"Theoretical ceiling (~0.82) due to 12% noise."* If your model tries to push its training loss to 0, it is memorizing the 12% random replacements. Smoothing acknowledges the uncertainty and keeps the model focused on the 88% true underlying logical rules.

---

## Summary Checklist for Maximum Score

- [x] Base model: `microsoft/deberta-v3-base` or `large`
- [x] Validation: Held-out `(class, environment)` pairs.
- [x] Loss Function: Multi-task Cross Entropy across all 8 columns + 2 auxiliary columns.
- [x] Regularization: High dropout, word masking, and label smoothing to combat 12% noise.
- [x] Final Output: Ensure all predictions map exactly to the allowed constants in `generate_data.py`.
