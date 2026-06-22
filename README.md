# Predicting Loan Default - A Machine Learning Approach to Credit Risk Assessment - 
End-to-end machine learning project that predicts loan defaults on the Lending Club dataset (2007–2018) using only information available at the time of loan origination. The project covers the full pipeline data cleaning, leakage-aware preprocessing, EDA, multiple models with hyperparameter tuning, and a business-driven threshold analysis.

**Author:** Atilla Ahmed
**Dataset:** Lending Club Loan Data (2007–2018) — ~2.26M loans, 145 features
**Best model:** Tuned `HistGradientBoostingClassifier` **ROC-AUC 0.733**, **PR-AUC 0.408**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Highlights](#project-highlights)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Problem Statement

Lending Club was the largest peer-to-peer lending platform in the US. The central risk for investors is **credit default** a borrower who stops repaying causes a partial or total loss of principal.

This project builds a binary classifier to predict default (`Charged Off` = 1) vs full repayment (`Fully Paid` = 0), using **only features available at loan origination**. Any post-origination variable is excluded to prevent data leakage and to ensure the model is usable on new applications.

The work is framed inside the standard credit-risk equation:

$$\text{Expected Loss} = PD \times LGD \times EAD$$

where the model targets the **PD (Probability of Default)** component.

---

## Project Highlights

- **Full sklearn `Pipeline` architecture** with `ColumnTransformer` and three custom transformers (`BaseEstimator` + `TransformerMixin`).
- **Strict leakage prevention** train/test split executed *before* any imputation, scaling, or rare-category grouping.
- **38 leakage columns identified and removed** (post-origination repayment fields).
- **Five models trained and compared:** Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting (default + tuned).
- **`RandomizedSearchCV`** with stratified 3-fold CV scoring on `average_precision`.
- **Permutation importance** used instead of impurity-based importance (unbiased and model-agnostic).
- **Business-driven threshold analysis** finds the operating point that maximises expected dollar profit, not raw F1.

---

## Dataset

| Item | Value |
|---|---|
| Source | [Kaggle — Lending Club 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Raw size | 2,260,668 rows × 145 columns (~1.1 GB) |
| After filtering to known outcomes | 1,306,387 rows |
| Final feature count after cleaning | ~55 numeric + categorical features |
| Default rate | ~20% |
| Target | `loan_status` → binary (`Charged Off` = 1, `Fully Paid` = 0) |

Rows with statuses `Current`, `Late`, or `In Grace Period` were excluded — their outcome is unknown.

---

## Methodology

### 1. Data Cleaning
- Filter to loans with definitive outcomes.
- Re-compute missing rates **after** filtering (since LC introduced new columns over time).
- Drop columns in four well-defined groups: high-missing (>50%), post-origination leakage, zero-variance, identifiers/free-text.
- Type conversions: `term`, `emp_length`, `earliest_cr_line` → numeric.
- Missing-value handling: binary "missingness flag" + median/mode imputation, with separate logic for `mths_since_*` columns (where missing means *never happened*).

### 2. Exploratory Data Analysis
- Univariate distributions of six key risk drivers.
- Bivariate analysis: defaulters vs non-defaulters.
- Categorical default-rate analysis (grade, sub-grade, term, purpose, etc.).
- Correlation heatmap → multicollinear pairs identified and removed.

### 3. Feature Engineering & Pipeline
- **Custom transformers:**
  - `RatioFeatureEngineer` - five financial ratios (installment/income, loan/income, etc.).
  - `SubGradeOrdinalEncoder` - maps `A1…G5` to a single ordinal value (35 levels).
  - `RareCategoryGrouper` - groups `addr_state` categories below 1% into `'other'`, fit on **train only**.
- **`ColumnTransformer`** applies median imputation + `StandardScaler` to numeric features and `OneHotEncoder` to nominal categoricals.
- All preprocessing fit on `X_train` only.

### 4. Modeling
| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear baseline |
| Random Forest | Variance-reduction ensemble |
| HistGradientBoosting | Bias-reduction ensemble |
| **HistGradientBoosting Tuned** | Final model `RandomizedSearchCV` over depth, learning rate, iterations, leaf size |

Class imbalance handled with `class_weight='balanced'`. Models compared on a held-out 20% test set.

---

## Results

| Model | ROC-AUC | PR-AUC | Recall (Default) | Precision (Default) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.718 | 0.381 | 0.65 | 0.33 |
| Decision Tree | 0.705 | 0.366 | 0.66 | 0.31 |
| Random Forest | 0.7194 | 0.388 | 0.637 | 0.33 |
| HistGradientBoosting | 0.726 | 0.399 | 0.68 | 0.33 |
| **HistGradientBoosting Tuned** | **0.733** | **0.408** | **0.68** | **0.33** |

## Key Insights

1. **`sub_grade` dominates.** Permutation importance shows removing it costs **0.103 PR-AUC** more than three times the next feature.
2. **Models converge.** All five models fall within a 0.042 PR-AUC band → the **data is the ceiling, not the algorithm**.
3. **Class overlap is fundamental.** The borrowers who default *look very similar* to those who don't at origination time. The most discriminative variables are behavioural and only appear after the loan is active.
4. **Boosting > Bagging here.** Bias reduction (HGB) marginally beats variance reduction (RF) consistent with the difficulty being underfitting, not overfitting.

---

## Limitations

**Limitations**
- Origination-time features cap performance behavioural data would unlock real gains.
- Severe class overlap means the precision/recall curve is steep; no model fully escapes it.
- Calibration was not formally evaluated; for production PD use, isotonic or Platt scaling should be added.

---

## Tech Stack

- **Python 3.10+**
- **pandas**, **NumPy** - data manipulation
- **scikit-learn** - pipelines, models, evaluation
- **matplotlib**, **seaborn** - visualisation
- **scipy** - hyperparameter search distributions
- **Jupyter Notebook**

---

## References

**Dataset**
- wordsforthewise. (2019). *All Lending Club loan data, 2007–2018* [Dataset]. Kaggle. <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

**Papers**
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

---
