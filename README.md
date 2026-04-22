# FABIP: Fairness-Aware Bias Impact Pruning

**A General Framework for Sustainable and Fair Neural Network Prediction**

> *"Can a model be made fairer and smaller at the same time — without sacrificing accuracy?"*  
> FABIP answers yes, through principled structured pruning guided by a mathematically justified Bias Impact Score.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Key Contributions](#key-contributions)
4. [Mathematical Framework](#mathematical-framework)
5. [Architecture](#architecture)
6. [Datasets](#datasets)
7. [Baselines](#baselines)
8. [Results](#results)
9. [Visualisations](#visualisations)
10. [Project Structure](#project-structure)
11. [How to Run](#how-to-run)
12. [Ablation Study](#ablation-study)
13. [Sensitivity Analysis](#sensitivity-analysis)
14. [Multi-Dataset Generalisation](#multi-dataset-generalisation)
15. [What "Sustainable" Means](#what-sustainable-means)
16. [Authors](#authors)
17. [References](#references)

---

## Overview

**FABIP (Fairness-Aware Bias Impact Pruning)** is a structured neural network pruning framework that simultaneously reduces demographic bias and computational cost in classification models.

Unlike traditional pruning methods, FABIP:
- Identifies **bias-inducing neurons** using a **Bias Impact Score (BIS)**
- Applies **structured pruning** (removes neurons, not just weights)
- Maintains model accuracy while improving fairness and efficiency

The framework is evaluated across:
- COMPAS (criminal justice)
- Adult Income (finance/hiring)
- German Credit (finance)

---

## Motivation

Machine learning systems often produce biased outcomes due to historical and societal inequalities embedded in data.

Existing approaches:
- Fairness constraints → reduce accuracy
- Standard pruning → ignores bias

**FABIP solves this by:**
- Detecting bias at neuron level
- Removing bias-causing neurons
- Preserving performance

---

## Key Contributions

| # | Contribution |
|---|---|
| C1 | Bias Impact Score (BIS) using masking + gradients |
| C2 | Fairness Pruning Score (FPS) balancing bias and importance |
| C3 | Structured pruning → real FLOPs reduction |
| C4 | Multi-dataset generalisation |
| C5 | Practical definition of sustainable ML |

---

## Mathematical Framework

### Bias Impact Score (BIS)

BIS is computed using **masking (ablation)**:

A neuron is temporarily removed (masked), and the change in prediction disparity across sensitive groups is measured. This captures the **causal contribution of that neuron to bias**.

### Importance

Measured using gradient magnitude:

I(j) = || ∂L / ∂h_j ||

### Fairness Pruning Score (FPS)

FPS(j) = λ1 * BIS(j) − λ2 * Importance(j)

High FPS → prune first

### Structured Pruning

Entire neurons are removed, reducing:
- Parameters
- FLOPs
- Latency

---

## Architecture

The baseline model is a 3-hidden-layer **MLP with BatchNorm and Dropout**:

```
Input (7 features)
    ↓
Linear(7 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(64 → 32)  + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(32 → 1)   [Binary output via BCEWithLogitsLoss]
```

After FABIP structured pruning at ratio 0.30, the architecture becomes physically smaller (exact dimensions depend on which neurons are identified as bias carriers — typically around `128 → 64 → 32` shrinks to `~90 → ~44 → ~22`).

**Training configuration:**
| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 (baseline), 5e-4 (fine-tune) |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=4, factor=0.5) |
| Early stopping patience | 10 epochs |
| Batch size | 256 |
| Random seed | 42 |

---

## Datasets

### Primary — COMPAS (Recidivism)

| Property | Detail |
|---|---|
| **Domain** | Criminal Justice |
| **Task** | Binary classification: predict recidivism within 2 years |
| **Sensitive attribute** | Race (African-American = 1, Other = 0) |
| **Features** | Age, prior convictions, juvenile felony/misdemeanour/other counts, charge degree |
| **Target** | `two_year_recid` (0 = did not reoffend, 1 = reoffended) |
| **Source** | ProPublica / [Kaggle — COMPAS Recidivism Racial Bias](https://www.kaggle.com/datasets/danofer/compass) |
| **Total samples** | ~6,172 (after filtering) |
| **Train / Val / Test** | 4,320 / 926 / 926 |
| **Known bias** | Black defendants are assigned higher risk scores than comparable White defendants |

**Dataset filtering applied** (ProPublica methodology):
- `days_b_screening_arrest` between −30 and +30
- `is_recid ≠ −1`
- `c_charge_degree ≠ 'O'`
- `score_text ≠ 'N/A'`

### Secondary — Adult Income (UCI)

| Property | Detail |
|---|---|
| **Domain** | Finance / Hiring |
| **Task** | Predict income > $50K |
| **Sensitive attribute** | Sex (Female = 1, Male = 0) |
| **Source** | UCI ML Repository via `sklearn.datasets.fetch_openml` |
| **Known bias** | Female workers systematically under-predicted for high-income class |

### Secondary — German Credit (UCI)

| Property | Detail |
|---|---|
| **Domain** | Finance / Credit |
| **Task** | Predict credit risk (good = 1, bad = 0) |
| **Sensitive attribute** | Age (Young < 25 = 1, Older = 0) |
| **Source** | UCI ML Repository via `sklearn.datasets.fetch_openml` |
| **Known bias** | Younger applicants assigned worse credit risk regardless of financial history |

---

## Baselines

Three baselines are implemented and compared against FABIP:

### 1. Vanilla (Unpruned)
A standard neural network trained with cross-entropy loss and no fairness intervention. Establishes the accuracy ceiling and the bias floor — the worst-case fairness and best-case efficiency.

### 2. Magnitude Pruning (Standard Unstructured)
The lowest-magnitude weights globally are zeroed using L1-based unstructured pruning (40% of weights). Fine-tuned afterward.

**Key limitation (why it appears in this study):** Zeroed weights are still stored in memory and still participate in matrix multiplication. The network dimensions are unchanged. **FLOPs do not decrease.** This is the most common mistake in the pruning-fairness literature and is explicitly demonstrated in this work.

### 3. FairPrune (Wu et al., 2022)
Neurons whose activations are most correlated with the sensitive attribute $A$ are zeroed out. This is the closest prior work to FABIP, applied originally to dermatological disease diagnosis.

**Key limitation:** Also unstructured — correlation-based zeroing does not change matrix dimensions, so FLOPs remain unchanged. Furthermore, it uses a single signal (correlation) rather than FABIP's dual-verification BIS.

### FABIP (Ours)
Structured pruning guided by the ensemble BIS + FPS. Physically removes neurons and rebuilds the network smaller. The only method in this comparison that achieves genuine reductions in FLOPs, parameters, and inference latency.

---

## Results

All results are on the **COMPAS test set** (926 samples), reported after fine-tuning.

### Main Results Table

| Model | Accuracy | F1 | AUC | DP ↓ | EO ↓ | DI → 1.0 | #Params | FLOPs | Latency (μs) |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.6771 | 0.6071 | 0.7176 | 0.2718 | 0.4598 | 2.1964 | 11,841 | 22,336 | 23.604 |
| Magnitude Pruning | — | — | — | — | — | — | 11,841 | 22,336 | — |
| FairPrune | — | — | — | — | — | — | 11,841 | 22,336 | — |
| **FABIP (Ours)** | **0.6782** | **0.6140** | **0.7209** | **0.2603** | **0.4340** | **2.0623** | **6,213** | **11,476** | **18.842** |

> Magnitude Pruning and FairPrune are included in the notebook with full metrics. Their FLOPs and #Params columns match Vanilla because both are **unstructured** — this is an intentional design point of the study.

### FABIP vs Vanilla — Summary of Improvements

| Metric | Vanilla | FABIP | Change |
|---|---|---|---|
| Demographic Parity ↓ | 0.2718 | 0.2603 | **−4.2% (fairer)** |
| Equalized Odds ↓ | 0.4598 | 0.4340 | **−5.6% (fairer)** |
| Disparate Impact → 1.0 | 2.1964 | 2.0623 | **Closer to 1.0** |
| Accuracy | 0.6771 | 0.6782 | **+0.1% (unchanged)** |
| F1-Score | 0.6071 | 0.6140 | **+0.7%** |
| AUC | 0.7176 | 0.7209 | **+0.5%** |
| FLOPs | 22,336 | 11,476 | **−48.6%** |
| Parameters | 11,841 | 6,213 | **−47.5%** |
| Inference latency | 23.604 μs | 18.842 μs | **−20.2%** |

The key finding: FABIP improves all three fairness metrics, maintains (and slightly improves) accuracy, and cuts computational cost nearly in half — none of which is achievable with unstructured pruning methods.

### Fairness Metric Definitions

| Metric | Formula | Interpretation |
|---|---|---|
| **Demographic Parity (DP)** | `|P(ŷ=1|A=0) − P(ŷ=1|A=1)|` | Difference in positive prediction rates between groups. Lower = fairer. |
| **Equalized Odds (EO)** | `|TPR_priv − TPR_unpriv| + |FPR_priv − FPR_unpriv|` | Combined difference in error rates between groups. Lower = fairer. |
| **Disparate Impact (DI)** | `P(ŷ=1|A=1) / P(ŷ=1|A=0)` | Ratio of positive prediction rates. Values close to 1.0 = fairer. |

---

## Visualisations

The notebook produces **8 publication-quality figures**, all saved as `.png` files:

| Figure | File | Description |
|---|---|---|
| **Fig 1** | `fig1_main_comparison.png` | 6-panel dashboard: Accuracy, DP, EO, DI, FLOPs, Parameters across all 4 methods |
| **Fig 2** | `fig2_scatter.png` | Fairness–Accuracy scatter plot (bubble size = FLOPs); EO vs F1 scatter |
| **Fig 3** | `fig3_bis_heatmap.png` | Per-neuron BIS heatmap (gradient vs ablation) + FPS bar chart with pruned/kept highlighted |
| **Fig 4** | `fig4_training_curves.png` | Validation loss and accuracy curves: Vanilla training vs FABIP fine-tuning |
| **Fig 5** | `fig5_group_accuracy.png` | Per-group accuracy (privileged vs unprivileged) and accuracy gap across all methods |
| **Fig 6** | `fig6_ablation.png` | Ablation study: effect of removing each FABIP component on accuracy and fairness |
| **Fig 7** | `fig7_sensitivity.png` | Sensitivity analysis: accuracy, DP, and FLOPs as pruning ratio varies from 10% to 50% |
| **Fig 8** | `fig8_multidataset.png` | Multi-dataset generalisation: FABIP vs Vanilla across COMPAS, Adult, German Credit |

---

## Project Structure

```
FABIP_Project.ipynb          ← Main notebook (Google Colab, GPU recommended)
README.md                    ← This file
compas-scores-two-years.csv  ← Primary dataset (upload to /content/ in Colab)
fig1_main_comparison.png     ← Generated after running Section 10
fig2_scatter.png             ← Generated after running Section 10
fig3_bis_heatmap.png         ← Generated after running Section 10
fig4_training_curves.png     ← Generated after running Section 10
fig5_group_accuracy.png      ← Generated after running Section 10
fig6_ablation.png            ← Generated after running Section 11
fig7_sensitivity.png         ← Generated after running Section 12
fig8_multidataset.png        ← Generated after running Section 13
```

---

## How to Run

### ▶️ Run Directly in Google Colab (Recommended)

👉 **[Open in Colab](https://colab.research.google.com/github/MNMashiur/FABIP-Fairness-Aware-Bias-Impact-Pruning-for-Sustainable-and-Fair-Recidivism-Prediction/blob/main/FABIP.ipynb)**

---

### Steps

1. Click the **Open in Colab** link above  
2. *(Optional)* Enable GPU for faster execution:  
   **Runtime → Change runtime type → T4 GPU**  
3. Click:  
   **Runtime → Run all**

---

### ✅ Ready to Go

- No dataset upload required  
- No setup required  
- Everything is pre-configured  

---

### ⏱ Runtime

- Approximately **20–40 minutes** on GPU  
- Includes:
  - Baseline training  
  - FABIP pruning  
  - Fairness evaluation  
  - Ablation study  
  - Sensitivity analysis  

---

### 🧠 Notes

- Outputs (metrics, plots, comparisons) are **already visible in GitHub preview**
- You can **rerun and modify everything directly in Colab**
- Designed for **one-click reproducibility**

---


### Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Model training, pruning, gradient computation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `scikit-learn` | ≥ 1.3 | Metrics, preprocessing, dataset loading |
| `matplotlib` | ≥ 3.7 | All visualisations |
| `seaborn` | ≥ 0.12 | Heatmap styling |
| `fairlearn` | ≥ 0.9 | Fairness metric reference |
| `fvcore` | ≥ 0.1 | FLOPs counting utility |

---

## Ablation Study

To rigorously justify each component of FABIP, three degraded variants are evaluated:

| Variant | BIS Used | Importance Used | Pruning Type | Purpose |
|---|---|---|---|---|
| No-BIS | ✗ | ✓ | Structured | Proves BIS is necessary — removing unimportant neurons alone does not improve fairness |
| No-Importance | ✓ | ✗ | Structured | Proves importance weighting is necessary — BIS-only pruning can hurt accuracy |
| No-Ablation | Gradient only | ✓ | Structured | Proves ablation verification adds value over gradient BIS alone |
| **FABIP (Full)** | Gradient + Ablation | ✓ | Structured | Best fairness with minimal accuracy cost |

All four variants use the same pruning ratio (0.30) for a fair comparison. Results confirm that every component contributes — removing any single element degrades either fairness or accuracy.

---

## Sensitivity Analysis

FABIP is evaluated across five pruning ratios to characterise the fairness–efficiency–accuracy trade-off:

| Pruning Ratio | Behaviour |
|---|---|
| 0.10 | Minimal fairness improvement; FLOPs barely change |
| 0.20 | Moderate fairness gains; small FLOPs reduction |
| **0.30** | **Best balance — chosen as the default** |
| 0.40 | Further FLOPs reduction; minor additional accuracy cost |
| 0.50 | Aggressive pruning; accuracy begins to degrade noticeably |

The chosen default of **30% pruning** achieves the best balance between fairness improvement, accuracy preservation, and computational reduction across all three datasets.

---


## Multi-Dataset Generalisation

A full pipeline (vanilla training → FABIP) is run on all three datasets independently. This confirms that FABIP is not overfit to the COMPAS dataset or recidivism prediction specifically, but is a general fairness-aware pruning framework.

**Summary of FABIP improvements over Vanilla across datasets:**

| Dataset | Sensitive Attr. | DP Improvement | EO Improvement | FLOPs Reduction |
|---|---|---|---|---|
| COMPAS | Race | −4.2% | −5.6% | ~48% |
| Adult Income | Sex | Consistent reduction | Consistent reduction | ~45–50% |
| German Credit | Age | Consistent reduction | Consistent reduction | ~45–50% |

Exact numbers for Adult and German Credit are printed in **Section 13** of the notebook.

---

## What "Sustainable" Means

The term **"sustainable"** in the project title is defined operationally and precisely throughout the code and this README. It is not used loosely.

A model is sustainable if it satisfies all three of the following:

1. **Fair** — demonstrates reduced demographic disparity (lower DP and EO) compared to an unpruned baseline
2. **Accurate** — maintains predictive performance within an acceptable threshold (< 2% accuracy loss in our experiments)
3. **Computationally efficient** — achieves genuine reductions in FLOPs, parameter count, and inference latency through structured (not unstructured) pruning

Standard unstructured pruning methods (including FairPrune) fail condition 3 because zeroing weights does not change the physical dimensions of weight matrices. Only FABIP's structured approach satisfies all three conditions simultaneously.

---

## Authors

| Name | Student ID | Contribution |
|---|---|---|
| Mashiur Rahaman Mollah Niran | 2022-3-60-183 | 40% |
| Nipu Basher | 2022-3-60-265 | 30% |
| Tasmia Rahman | 2022-3-60-164 | 30% |

**Submitted To:** Ahmed Abdal Shafi Rasel, Senior Lecturer, Department of CSE, East West University

---

## References

## References

[1] J. Angwin, J. Larson, S. Mattu, and L. Kirchner, “Machine bias,” ProPublica, May 2016. https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

[2] K. Dastin, “Amazon scraps secret AI recruiting tool that showed bias against women,” Reuters, Oct. 2018. https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G

[3] L. Thomas, J. Crook, and D. Edelman, *Credit Scoring and Its Applications*. SIAM, 2017.

[4] Z. Obermeyer et al., “Dissecting racial bias in an algorithm used to manage the health of populations,” *Science*, 2019.

[5] M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” *NeurIPS*, 2016.

[6] F. Kamiran and T. Calders, “Data preprocessing techniques for classification without discrimination,” *KAIS*, 2012.

[7] B. H. Zhang et al., “Mitigating unwanted biases with adversarial learning,” *AIES*, 2018.

[8] G. Pleiss et al., “On fairness and calibration,” *NeurIPS*, 2017.

[9] S. Han et al., “Learning both weights and connections for efficient neural networks,” *NeurIPS*, 2015.

[10] Y. LeCun et al., “Optimal brain damage,” *NeurIPS*, 1989.

[11] Y. Wu et al., “FairPrune,” *MICCAI*, 2022.

[12] E. Iofinova et al., “Bias in pruned vision models,” *CVPR*, 2023.

[13] N. Mehrabi et al., “A survey on bias and fairness in machine learning,” *ACM CSUR*, 2021.

[14] A. Chouldechova, “Fair prediction with disparate impact,” *Big Data*, 2017.

[15] S. Barocas, M. Hardt, and A. Narayanan, *Fairness and Machine Learning*, MIT Press, 2023.

[16] J. Kleinberg et al., “Human decisions and machine predictions,” *QJE*, 2018.

[17] J. Dressel and H. Farid, “The accuracy, fairness, and limits of predicting recidivism,” *Science Advances*, 2018.

[18] B. Jain et al., “Singular race models,” *PETRA*, 2019.

[19] P. Kondapalli et al., “Bias detection in criminal justice,” *Engineering Proceedings*, 2025.

[20] M. Feldman et al., “Certifying and removing disparate impact,” *KDD*, 2015.

[21] L. E. Celis et al., “Classification with fairness constraints,” *FAT*, 2019.

[22] T. P. Pagano et al., “Bias and unfairness in ML models,” *BDCC*, 2023.

[23] D. Li, “Pruning filters for efficient convnets,” arXiv, 2016.

[24] P. Molchanov et al., “Pruning CNNs for efficient inference,” arXiv, 2016.

[25] Y. He et al., “Soft filter pruning,” arXiv, 2018.

[26] J. Frankle and M. Carbin, “Lottery Ticket Hypothesis,” arXiv, 2018.

[27] A. Zayed et al., “Fairness-aware structured pruning,” *AAAI*, 2024.

[28] L. Zhang et al., “Fairness-aware adversarial pruning,” *ICCV*, 2023.

[29] Y. Dai et al., “Fairness and pruning via bi-level optimization,” arXiv, 2023.

[30] H. Kasyap et al., “Mitigating bias via pruning,” *ECAI*, 2024.

[31] Q. Qin and E. Merlo, “Prune bias from the root,” *IST*, 2026.

[32] Y. Xue et al., “BMFT: Bias-based masking fine-tuning,” *MICCAI Workshop*, 2024.

[33] D. Kingma and J. Ba, “Adam optimizer,” arXiv, 2014.

[34] P. Koh and P. Liang, “Influence functions,” *ICML*, 2017.

[35] R. Kohavi, “Scaling up Naive Bayes,” *KDD*, 1996.

[36] H. Hofmann, “German Credit dataset,” UCI Repository, 1994.

[37] S. Lundberg and S. Lee, “SHAP values,” *NeurIPS*, 2017.

[38] T. Chen et al., “Contrastive learning framework,” *ICML*, 2020.

[39] M. Shah and N. Sureja, “Bias in deep learning models,” *ACM Eng.*, 2025.

[40] Z. Chen et al., “Bias mitigation methods study,” *TOSEM*, 2023.

[41] T. Jui and P. Rivas, “Fairness issues in ML,” *IJMLC*, 2024.

[42] R. Rabonato and L. Berton, “Fairness systematic review,” *AI Ethics*, 2025.

[43] A. Paszke et al., “PyTorch,” *NeurIPS*, 2019.

[44] F. Pedregosa et al., “Scikit-learn,” *JMLR*, 2011.

---

*Department of Computer Science and Engineering, East West University — 2026*
