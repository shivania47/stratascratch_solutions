# Starbucks Promotional Offer Experiment — Part 1: Experiment Analysis

> **StrataScratch Case Study** | A/B Test Analysis Pipeline built with [Marimo](https://marimo.io/)

---

## Background

This is a sample data project on Stratascratch. The data project is available here: https://platform.stratascratch.com/data-projects/which-customers-target

Question/Prompt: 

Starbucks ran a randomized experiment to evaluate whether an advertising promotion would increase purchases of a product priced at **$10**. Because each promotion costs **$0.15** to send, the business goal is not just to maximize conversions — it's to maximize **net incremental revenue** by targeting only the most receptive customers.

This notebook addresses **Part 1** of the case study: end-to-end experiment validation and analysis.


---

## Key Questions Answered

1. **Was the experiment valid?** — Full pre-analysis validity pipeline
2. **What did the experiment find?** — Average Treatment Effect (ATE), confidence intervals, and business impact

---

## Dataset

**File:** `datasets/Training.csv`

| Column | Description |
|---|---|
| `ID` | Customer identifier |
| `Promotion` | Treatment assignment (`Yes` = received promotion, `No` = control) |
| `purchase` | Binary outcome (1 = purchased, 0 = did not purchase) |
| `V1` – `V7` | Pre-treatment customer features (mix of categorical and continuous) |

**Cost parameters:**
- Promotion cost (treatment): **$0.15 per customer**
- Revenue per purchase: **$10.00**

---

## Metrics

| Metric | Definition |
|---|---|
| **Incremental Response Rate (IRR)** | Purchase rate in treatment minus purchase rate in control |
| **Net Incremental Revenue (NIR)** | `10 × (treatment purchasers) − 0.15 × (promotions sent) − 10 × (control purchasers)` |

---

## Analysis Pipeline

### Section 1a — Experiment Validation

| Check | Function | Result |
|---|---|---|
| Randomization Integrity | `check_overlap()` 
| ID Deduplication | `check_id_duplication()` 
| Value Distribution | `check_value_distribution()`
| Covariate Distribution | `check_covariate_distribution()` 
| Covariate Balance | `check_covariate_balance()` 
| Sample Ratio Mismatch | `check_sample_ratio_mismatch()` 
| Minimum Detectable Lift | `check_minimum_detectable_lift()` 

### Section 1b — Experiment Analysis

The `AnalyzeExperiment` class runs the full statistical evaluation:

- **Point estimates** via `compute_metrics()` — ATE and relative lift for all metrics
- **Statistical tests** via `compute_statistics()` — z-test (conversion rate), t-test + bootstrap CI (revenue)
- **Business impact** via `compute_nir()` and `project_impact()`

---

## Executive Summary

**Part 1a — Validity:** The experiment passed all validation checks. No user overlap, no duplicate IDs, no sample ratio mismatch, and pre-treatment covariates are balanced across treatment arms.

**Part 1b — Results:**

1. The promotion drove a **statistically significant lift in conversion rate** (IRR).
2. However, because only a small subset of users converted, the **cost of promotion outweighed the incremental revenue**, resulting in a **negative Net Incremental Revenue** overall.
3. **This indicates the promotion should be targeted** — sent only to customers most likely to respond — in order to maximize NIR. This motivates Part 2 of the analysis (uplift modeling) which is coming soon! 

---

## Tech Stack

| Library | Purpose |
|---|---|
| `marimo` | Reactive notebook framework |
| `pandas`, `numpy` | Data manipulation |
| `scipy`, `statsmodels` | Statistical tests (z-test, t-test, KS, Chi², bootstrap) |
| `matplotlib`, `seaborn` | Visualization |

## Author

**Shivania** | [GitHub](https://github.com/shivania47)

*Solution developed as part of the StrataScratch data science challenge series.*
