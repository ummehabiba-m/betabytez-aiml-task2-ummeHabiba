# Model Training on Two Datasets with Comparison
## 📌 Overview

This project trains and compares classification models across **two datasets from
different domains**, then reports on how data quality and structure affect model
performance.

| | Dataset 1 (Medical) | Dataset 2 (Financial) |
|---|---|---|
| **Name** | Pima Indians Diabetes | Loan Approval Prediction |
| **Task** | Predict diabetes onset (0/1) | Predict loan approval (0/1) |
| **Rows** | 768 | 4,269 |
| **Features** | 8, all numeric | 12 (numeric + 2 categorical: education, self_employed) |
| **Quirks** | Missing values hidden as `0` (Glucose, BMI, etc.) | No missing values; one dominant predictor (`cibil_score`) |
| **Source** | [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | Kaggle "Loan Approval Prediction Dataset" |

## 🗂️ Structure

```
betabytez-aiml-task2-ummehabiba/
├── data/
│   ├── pima_diabetes.csv
│   └── loan_prediction.csv
├── notebooks/
│   └── Task2_Model_Comparison.ipynb   # full analysis, models, comparison report
├── requirements.txt
└── README.md
```

## 🔑 Key Finding

Data quality set the performance ceiling far more than model choice or tuning did.
**Random Forest was the stronger model on both datasets** (F1 = 0.653 on Diabetes
vs. F1 = 0.984 on Loan Approval) — but that 0.33-point gap came entirely from the
data, not the algorithm. The Diabetes dataset had disguised missing values, a
weaker feature-target signal, and a smaller sample (768 rows); the Loan Approval
dataset was clean, had a near-deterministic predictor (`cibil_score`), and had
~6x more rows. Improvement techniques (`GridSearchCV`, `SMOTE`) helped incrementally
within each dataset, but couldn't close the gap *between* datasets of different
underlying quality — the practical lesson being that cleaning/understanding data
is often higher-leverage than model selection or tuning.

## 📊 Results Summary

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | Diabetes | 0.708 | 0.600 | 0.500 | 0.545 |
| Random Forest | Diabetes | 0.779 | 0.727 | 0.593 | **0.653** |
| Logistic Regression (Tuned, GridSearchCV) | Diabetes | 0.708 | 0.571 | 0.667 | 0.615 |
| Logistic Regression | Loan Approval | 0.913 | 0.921 | 0.942 | 0.931 |
| Random Forest | Loan Approval | 0.980 | 0.983 | 0.985 | **0.984** |
| Logistic Regression (SMOTE) | Loan Approval | 0.923 | 0.951 | 0.923 | 0.937 |

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Task2_Model_Comparison.ipynb
```
**Author:** Umme Habiba Malik
**Track:** BetaBytez AI/ML Internship — Week 2
