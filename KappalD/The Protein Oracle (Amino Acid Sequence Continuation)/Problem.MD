# 🧬 Protein Sequence Continuation Challenge

![Biology](https://img.shields.io/badge/Domain-Proteomics-green) ![Task](https://img.shields.io/badge/Task-Sequence_Prediction-blue) ![Constraints](https://img.shields.io/badge/Parameters-%3C50M-red)

## 1. Challenge Overview & Objective

### I. Real World Context
Proteins are the functional building blocks of life. Each protein's behavior is determined by its amino acid sequence. Predicting how a sequence continues requires understanding deep biochemical patterns: hydrophobic core packing, secondary structure propensities ($\alpha$-helix vs. $\beta$-sheet), domain boundaries, and conserved functional motifs.

### II. Objective
Given the first **50-75%** of a human protein's amino acid sequence, you must predict the exact **next 20 residues**. Your model must learn the "grammar" of protein sequences from scratch:
* **Local constraints:** How residues choice is limited by secondary structures.
* **Long-range patterns:** How repeating motifs influence downstream selection.

### III. The Training Mandate
* **Model Constraint:** You must train a sequence model from scratch under **50 Million total parameters**.
* **Pre-trained Weights Prohibition:** Using pre-trained weights from any general-purpose language model or protein foundation model (e.g., ESM, ProtTrans, ProGen) is **strictly prohibited**. 

---

## 2. Dataset Structure

The raw human proteome sequences are split at a random point. The prefix becomes the `context`, and the following 20 residues are the `continuation`.

### File Details

| File | Column | Data Type | Description |
|:---|:---|:---|:---|
| **train.csv** | `seq_id` | String | Unique 12-character identifier. |
| | `context` | String | Amino acid prefix (Avg. ~200-300 residues). |
| | `continuation`| String | The next 20 residues (**Target**). |
| **test.csv** | `seq_id` | String | Unique identifier. |
| | `context` | String | Prefix for prediction. |
| **sample_submission.csv** | `seq_id` | String | Matching identifier from `test.csv`. |
| | `continuation`| String | Placeholder (default: `AAAA...`). |

---

## 3. Evaluation Metric

The competition uses **Per-Residue Positional Accuracy**. Each of the 20 predicted positions is compared character-by-character against the ground truth.

$$Accuracy = \frac{\text{Number of correct residues at correct positions}}{\text{Total positions across all samples}}$$

* **Baseline:** The random baseline is approximately **5%** (1 out of 20 possible amino acids).
* **Penalty:** Predictions shorter than 20 residues result in incorrect counts for missing positions.

---

## 4. Source Data: Human Proteome Triplets

The dataset is derived from the `human_proteome_triplets` corpus (Hugging Face: `yarongef/human_proteome_triplets`).

### The Amino Acid Alphabet
Protein sequences consist of a strict 20-character alphabet. **Note:** Standard English BPE tokenizers often fragment these sequences in ways that destroy biochemical meaning.

| Letter | Amino Acid | Property | Letter | Amino Acid | Property |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Alanine | Hydrophobic | **K** | Lysine | Positive charge |
| **R** | Arginine | Positive charge | **M** | Methionine | Hydrophobic |
| **N** | Asparagine | Polar | **F** | Phenylalanine | Aromatic |
| **D** | Aspartic acid | Negative charge | **P** | Proline | Rigid/Breaker |
| **C** | Cysteine | Disulfide bonds | **S** | Serine | Polar |
| **E** | Glutamic acid | Negative charge | **T** | Threonine | Polar |
| **Q** | Glutamine | Polar | **W** | Tryptophan | Aromatic |
| **G** | Glycine | Small/Flexible | **Y** | Tyrosine | Aromatic |
| **H** | Histidine | Positive charge | **V** | Valine | Hydrophobic |
| **I** | Isoleucine | Hydrophobic | **L** | Leucine | Hydrophobic |

---

## 5. Submission Format

Your final output should be a CSV file saved to `./working/submission.csv` with exactly 2 columns:

| seq_id | continuation |
| :--- | :--- |
| `prot_a4b9c1` | `LGKSDFHWIERTQNMCVPAY` |
| `prot_ff91a2` | `AAMVLLCFHTRSEQGNIKDW` |

```python
import pandas as pd  
  
# Example submission export
submission = pd.DataFrame({  
    'seq_id': test_df['seq_id'],  
    'continuation': generated_continuations  
})  
  
submission.to_csv('submission.csv', index=False)
