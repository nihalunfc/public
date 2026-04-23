### 🧠 Advanced Optimization & Strategy Checklist
When analyzing the attached machine learning problem or proposing a solution, DO NOT default to a standard, baseline implementation (e.g., a simple Seq2Seq model with default Cross-Entropy loss). You must explicitly evaluate the problem against the following advanced techniques and recommend implementations where applicable:

#### 1. Advanced Ensembling & Stacking
* **Heterogeneous Averaging:** Can we train fundamentally different architectures (e.g., Encoder-Decoder, Decoder-only LLM, and Statistical/Markov models) and average their probability distributions to reduce variance?
* **Candidate Bank & Reranker:** Should the generative model output the top-K candidates, allowing a secondary, lightweight model (like Ridge Regression or Gradient Boosting) to score and select the final prediction based on engineered features (entropy, repetition, agreement)?

#### 2. Domain Priors & Logit Biasing
* **Physics/Business Constraints:** What domain rules govern this data? Explicitly define how to inject this logic directly into the pre-softmax logits (e.g., adding a positive bias to contextually frequent tokens, or applying negative infinity masks to physically/logically impossible predictions).

#### 3. Focal Grading & Custom Loss
* **Targeted Penalties:** Does the competition or business problem care more about a specific subset of the prediction? Modify the loss function to apply heavy multipliers (e.g., 5x weight) to the exact target zone (like the final 20 tokens) and ignore or lightly weight the prefix context. 
* **Regularization:** Implement Label Smoothing to prevent the model from becoming overly confident and brittle.

#### 4. Dynamic Data Augmentation
* **Synthetic Prefixing / Truncation:** How can we mathematically expand the training data? Propose dynamic dataloader functions that randomly slice, truncate, or mask the input context sequences during training to force the model to learn robust, position-independent patterns rather than memorizing sequence lengths.

**Output Directive:** For each of the four categories above, explicitly state whether it is applicable to the current problem. If yes, provide the technical strategy or code snippet to implement it. If no, briefly explain why it does not fit the problem domain.
