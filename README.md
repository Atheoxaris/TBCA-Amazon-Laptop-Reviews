# TBCA — Text-Based Conjoint Analysis
## Empirical Proof-of-Concept: Amazon Laptop Reviews

> **Paper:** *Bridging Decision Theory and Machine Learning: A Unified Framework for Consumer Preference Estimation from Unstructured Text*  
> **Authors:** Theoxaris Anastasiadis & Dimitrios Ampeliotis  
> **Institution:** Department of Digital Media and Communication, Ionian University, Greece  
> **Version:** April 2026

---

## Overview

This repository contains the empirical proof-of-concept for the **Text-Based Conjoint Analysis (TBCA)** framework introduced in the paper. The pipeline extracts consumer preference utilities directly from unstructured product reviews, without requiring pre-specified conjoint attributes or controlled experimental designs.

The implementation validates three core theoretical equivalences established in the paper:

| Utility Theory | Machine Learning Counterpart |
|---|---|
| MNL choice probability (RUM) | Softmax activation |
| Maximum Likelihood Estimation | Cross-entropy loss minimisation |
| MAUT additive utility structure | Linear network layer |

---

## Pipeline

```
Amazon Reviews (UGC)
        ↓
ABSA — DeBERTa-v3-base (SemEval-2016 Laptop14)
        ↓
Aspect-sentiment scores sᵢⱼ ∈ [−1, +1]
        ↓
Prospect Theory Debiasing (λ=2.25, α=0.88)
        ↓
MAUT Utility Estimation (Ridge regression)
        ↓
Softmax / MNL Choice Probabilities
        ↓
Axiomatic Consistency Verification
```

---

## Results

### Part-Worth Utility Weights (n=200 reviews, stratified sample)

| Attribute | Raw w̃ᵢ | Debiased ŵᵢ | Δ | 95% CI |
|---|---|---|---|---|
| Battery life    | 0.1279 | 0.0199 | −0.1080 ↓ | [0.005, 0.391] |
| Display quality | 0.0339 | 0.1747 | +0.1408 ↑ | [0.012, 0.439] |
| Performance     | 0.0469 | 0.2836 | +0.2368 ↑ | [0.011, 0.374] |
| Build quality   | 0.4112 | 0.2454 | −0.1658 ↓ | [0.025, 0.424] |
| Value for money | 0.3802 | 0.2763 | −0.1038 ↓ | [0.045, 0.484] |

**R² (CV-5):** 0.600 (raw) · 0.563 (debiased)  
**Bootstrap 95% CI** based on B=1,000 resamples (n=186 reviews with ≥1 mapped aspect).  
Wide intervals reflect the modest sample size; replication on larger corpora is warranted.

### Softmax Choice Probabilities

| Alternative | V̂ | P(choice) |
|---|---|---|
| Laptop A (Premium) | 0.6695 | 34.2% |
| Laptop B (Performance) | 0.6984 | **35.2%** |
| Laptop C (Budget) | 0.5618 | 30.7% |

### Axiomatic Consistency: 4/4 PASS ✓

- Σ P(choice) = 1.000000 — RUM probability axiom
- Σ ŵᵢ = 1.000000 — MAUT normalisation
- All Pⱼ ∈ (0,1) — RUM interior requirement
- All ŵᵢ > 0 — MAUT monotonicity

---

## Repository Structure

```
TBCA-Amazon-Laptop-Reviews/
│
├── TBCA_Colab_Notebook.ipynb   # Complete pipeline (Google Colab)
├── TBCA_Colab_Notebook.py      # Complete pipeline (Python file)
├── tbca_results.json           # Structured results
├── tbca_tables.tex             # LaTeX tables for paper
├── tbca_results.pdf            # Figures (300 dpi)
├── tbca_results.png            # Figures (PNG)
└── README.md
```

---

## Replication

### Option 1 — Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m7Oa7v75za3xkVeOHWPME_9UG0yguNfX)

1. Open the notebook via the badge above
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run cells sequentially (Cell 1 → Cell 13)
4. Estimated time: ~20 minutes

### Option 2 — Local execution

```bash
git clone https://github.com/Atheoxaris/TBCA-Amazon-Laptop-Reviews
cd TBCA-Amazon-Laptop-Reviews
pip install transformers torch pandas numpy scikit-learn matplotlib seaborn tabulate kaggle
jupyter notebook TBCA_Colab_Notebook.ipynb
```

### Dataset

The notebook downloads automatically via the Kaggle API:
- **Dataset:** [Amazon Laptop Review](https://www.kaggle.com/datasets/ashwinishet/amazon-laptop-review)
- **Source:** Kaggle (public, license: unknown)
- **Size:** 511 reviews · 3 columns (comments, rating, title)
- **Working sample:** 200 reviews (stratified random sample preserving star-rating distribution, `random_state=42`)

A Kaggle account and API token are required. See Cell 2 of the notebook for setup instructions.

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | DeBERTa-v3 ABSA model |
| `torch` | GPU inference |
| `pandas` / `numpy` | Data processing |
| `scikit-learn` | Ridge regression, cross-validation |
| `matplotlib` / `seaborn` | Figures |
| `kaggle` | Dataset download |
| `tabulate` | Results formatting |

---

## ABSA Model

**Model:** [`yangheng/deberta-v3-base-absa-v1.1`](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)  
**Architecture:** DeBERTa-v3-base  
**Fine-tuned on:** SemEval-2016 Task 5 — Laptop14 benchmark (Pontiki et al., 2016)  
**Technique:** Aspect prompting — input format: `aspect [SEP] review text`

---

## Theoretical Background

The TBCA framework rests on three pillars of legitimacy:

**Ontological:** Debreu's representation theorem guarantees that preferences expressed in text admit a continuous utility representation under completeness, transitivity, and continuity.

**Structural:** The MAUT additive utility form maps directly onto the linear layers of neural networks.

**Statistical:** The RUM and its MLE (equivalent to cross-entropy minimisation) guarantee consistent parameter estimation.

For the full theoretical development see the paper.

---

## Citation

```bibtex
@article{anastasiadis2026tbca,
  title   = {Bridging Decision Theory and Machine Learning: 
             A Unified Framework for Consumer Preference 
             Estimation from Unstructured Text},
  author  = {Anastasiadis, Theoxaris and Ampeliotis, Dimitrios},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## Contact

**Theoxaris Anastasiadis**  
PhD Candidate, Department of Digital Media and Communication  
Ionian University, Kefalonia, Greece  
[Atheoxaris@ionio.gr](mailto:Atheoxaris@ionio.gr)  
[github.com/Atheoxaris](https://github.com/Atheoxaris)
