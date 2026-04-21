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
16. [Addressing Faculty Feedback](#addressing-faculty-feedback)
17. [Authors](#authors)
18. [References](#references)

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

3-layer MLP:

Input → 128 → 64 → 32 → Output

With:
- BatchNorm
- ReLU
- Dropout

---

## Datasets

| Dataset | Domain | Sensitive Attribute |
|--------|--------|--------------------|
| COMPAS | Criminal Justice | Race |
| Adult Income | Finance | Sex |
| German Credit | Finance | Age |

---

## Baselines

- Vanilla (no pruning)
- Magnitude Pruning (unstructured)
- FairPrune (bias correlation-based)
- **FABIP (ours)**

---

## Results

FABIP achieves:

- Lower DP and EO (fairer predictions)
- Comparable or improved accuracy
- ~45–50% reduction in FLOPs and parameters

---

## Visualisations

Generated in notebook:
- Fairness vs accuracy plots
- BIS heatmaps
- Training curves
- Sensitivity analysis graphs

---

## Project Structure

FABIP_Project.ipynb  
README.md  
compas-scores-two-years.csv  
report.pdf  

---

## How to Run

### Google Colab

1. Open notebook in Colab  
2. Enable GPU (optional)  
3. Run all cells  

### Requirements

- torch  
- numpy  
- pandas  
- scikit-learn  

---

## Ablation Study

Evaluates the contribution of:
- BIS
- Importance
- Structured pruning

Shows that removing any component degrades performance.

---

## Sensitivity Analysis

Different pruning ratios tested (10%–50%).

Best balance achieved at ~30% pruning.

---

## Multi-Dataset Generalisation

FABIP consistently improves fairness across:
- COMPAS
- Adult Income
- German Credit

---

## What "Sustainable" Means

A model is **sustainable** if it is:

- Fair (reduced bias)
- Accurate (minimal performance loss)
- Efficient (lower FLOPs and latency)

---


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

1. Kasyap, H., et al. (2024). Mitigating Bias: Model Pruning for Enhanced Model Fairness and Efficiency. *ECAI 2024*, pp. 995–1002.
2. Wu, Y., et al. (2022). FairPrune: Achieving Fairness through Pruning for Dermatological Disease Diagnosis. *MICCAI 2022*, pp. 743–753.
3. Zayed, A., et al. (2024). Fairness-aware Structured Pruning in Transformers. *AAAI 2024*, Vol. 38, No. 20, pp. 22484–22492.
4. Zhang, L., et al. (2023). Towards Fairness-Aware Adversarial Network Pruning. *ICCV 2023*, pp. 5168–5177.
5. Dai, Y., et al. (2023). Integrating Fairness and Model Pruning Through Bi-level Optimization. *arXiv:2312.10181*.
6. Iofinova, E., et al. (2023). Bias in Pruned Vision Models: In-depth Analysis and Countermeasures. *CVPR 2023*, pp. 24364–24373.
7. Qin, Q., & Merlo, E. (2026). Prune Bias from the Root. *Information and Software Technology*, 189, 107906.
8. Kondapalli, P., et al. (2025). A Literature Review: Bias Detection and Mitigation in Criminal Justice. *Engineering Proceedings*, 107(1), 72.
9. Pagano, T. P., et al. (2023). Bias and Unfairness in Machine Learning Models. *BDCC*, 7(1), 15.
10. Chen, Z., et al. (2023). A Comprehensive Empirical Study of Bias Mitigation Methods. *ACM TOSEM*, 32(4), 1–30.
11. Xue, Y., et al. (2024). BMFT: Achieving Fairness via Bias-based Weight Masking Fine-tuning. *MICCAI Workshop on Fairness of AI in Medical Imaging*, pp. 98–108.
12. Jain, B., et al. (2019). Singular Race Models: Addressing Bias and Accuracy in Predicting Prisoner Recidivism. *PETRA 2019*, pp. 599–607.
13. Shah, M., & Sureja, N. (2025). A Comprehensive Review of Bias in Deep Learning Models. *Archives of Computational Methods in Engineering*, 32(1), 255–267.

---

*Department of Computer Science and Engineering, East West University — 2026*
