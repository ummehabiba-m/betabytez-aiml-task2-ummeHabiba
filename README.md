# 🩺💰 Model Training on Two Datasets with Comparison

**Author:** Umme Habiba Malik
**Program:** BetaBytez AI/ML Internship — Week 2
**Difficulty:** Beginner–Intermediate
**Tech Stack:** Python · Pandas · Scikit-learn · Matplotlib · Seaborn · imbalanced-learn

---

## 📌 Problem Statement

Data scientists rarely work with a single clean dataset. This project trains and
evaluates classification models on **two datasets from different domains** —
one medical, one financial — to understand *why* the same modeling approach
performs differently depending on the data it's given, and to communicate
that difference clearly through a written comparative analysis.

---

## 🗂️ Datasets

| | Dataset 1 — Medical 🩺 | Dataset 2 — Financial 💰 |
|---|---|---|
| **Name** | Pima Indians Diabetes | Loan Approval Prediction |
| **Goal** | Predict diabetes onset (0 = No, 1 = Yes) | Predict loan approval (0 = Rejected, 1 = Approved) |
| **Rows** | 768 | 4,269 |
| **Features** | 8, all numeric (Glucose, BMI, Age, etc.) | 12: 10 numeric + 2 categorical (`education`, `self_employed`) |
| **Data quality** | Missing values *disguised* as `0` in 5 columns | No missing values — clean out of the box |
| **Standout feature** | Glucose & BMI (moderate correlation with outcome) | `cibil_score` (dominant, near-deterministic predictor) |
| **Class balance** | ~65% No Diabetes / 35% Diabetes | ~62% Approved / 38% Rejected |
| **Source** | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | Kaggle "Loan Approval Prediction Dataset" |

---

## 🗂️ Repository Structure

```
betabytez-aiml-task2-ummehabiba/          ← this is the repo root
├── data/
│   ├── pima_diabetes.csv
│   └── loan_prediction.csv
├── notebook
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

For **each** dataset, the notebook follows the same pipeline:

1. **Load & Inspect** — shape, dtypes, summary statistics
2. **Preprocessing**
   - Dataset 1: detect disguised missing values (`0`s in Glucose/BMI/etc.), median-impute
   - Dataset 2: strip whitespace artifacts, encode categorical columns (`education`, `self_employed`, target)
3. **Exploratory Data Analysis** — 3 plots per dataset:
   - Feature distributions (histograms + KDE)
   - Correlation heatmap
   - Class balance bar chart
4. **Modeling** — two models trained per dataset:
   - Logistic Regression (interpretable linear baseline)
   - Random Forest (non-linear ensemble)
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix for every model
6. **Improving the weaker model** — a different technique per dataset, to demonstrate breadth:
   - Dataset 1 → `GridSearchCV` (5-fold CV hyperparameter search)
   - Dataset 2 → `SMOTE` (synthetic oversampling of the minority class)
7. **Comparative Analysis Report** — written in markdown cells at the end of the notebook

---

## 📊 Results Summary

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | Diabetes | 0.708 | 0.600 | 0.500 | 0.545 |
| Random Forest | Diabetes | 0.779 | 0.727 | 0.593 | **0.653** |
| Logistic Regression (Tuned — GridSearchCV) | Diabetes | 0.708 | 0.571 | 0.667 | 0.615 |
| Logistic Regression | Loan Approval | 0.913 | 0.921 | 0.942 | 0.931 |
| Random Forest | Loan Approval | 0.980 | 0.983 | 0.985 | **0.984** |
| Logistic Regression (SMOTE) | Loan Approval | 0.923 | 0.951 | 0.923 | 0.937 |

---

## 🔑 Key Findings

**1. Which dataset was harder to model, and why?**
Diabetes was clearly harder — its best F1-score (0.653) is far below Loan
Approval's best (0.984). Three reasons: disguised missing data requiring
imputation, no single dominant predictive feature, and a much smaller sample
(768 vs. 4,269 rows).

**2. Which model generalized better across both datasets?**
**Random Forest** — it won on every metric, on both datasets, without
exception. It captures non-linear feature interactions that Logistic
Regression's linear decision boundary misses.

**3. What does this say about data quality vs. model performance?**
**Data quality set the performance ceiling far more than model choice or
tuning did.** The same algorithm (Random Forest), with comparable effort,
scored 0.33 F1-points apart purely because of the underlying data. Cleaning
and understanding data is often a higher-leverage activity than model
selection or hyperparameter tuning.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run all cells top to bottom — the notebook loads both CSVs from the repo
root, so no path changes are needed if you clone the repo as-is.

---

## 🛠️ Tech Stack

- **pandas / numpy** — data loading & manipulation
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — Logistic Regression, Random Forest, GridSearchCV, metrics
- **imbalanced-learn** — SMOTE oversampling

---

