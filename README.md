# Task 1: Data Cleaning & Preprocessing
**AI & ML Internship — Elevate Labs**

## Objective
Learn how to clean and prepare raw data for machine learning.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Dataset
Titanic dataset (`data/titanic.csv`).

> **Note on the data file:** this dataset was generated locally (see
> `generate_dataset.py`) to match the real Kaggle Titanic dataset's exact
> columns, distributions, missing-value pattern, and outliers, since the
> environment this was built in had no internet access to download the
> original file. If you'd rather use the real file, download it from
> [Kaggle's Titanic competition](https://www.kaggle.com/competitions/titanic/data)
> and replace `data/titanic.csv` — `task1_preprocessing.py` needs no changes,
> since the column names match exactly.

## Project Structure
```
├── data/
│   ├── titanic.csv              # raw dataset
│   └── titanic_cleaned.csv      # final cleaned & processed dataset
├── images/
│   ├── missing_values_heatmap.png
│   ├── boxplots_before_outlier_removal.png
│   └── boxplots_after_outlier_removal.png
├── generate_dataset.py          # creates the raw dataset
├── task1_preprocessing.py       # full cleaning & preprocessing pipeline
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python generate_dataset.py       # only needed if data/titanic.csv doesn't exist
python task1_preprocessing.py
```

## What Was Done

1. **Explored the raw data** — checked shape, data types, and missing values
   with `.info()`, `.isnull().sum()`, and `.describe()`. Visualized missingness
   with a heatmap.
2. **Handled missing values**
   - `Age` (numeric, ~20% missing) → filled with the **median** (robust to
     skew/outliers, unlike the mean).
   - `Embarked` (categorical, 2 missing) → filled with the **mode**.
   - `Cabin` (~80% missing) → too sparse to impute reliably, so instead of
     dropping the information entirely it was turned into a binary
     `Has_Cabin` feature, then the original column was dropped.
   - Dropped `PassengerId`, `Name`, `Ticket` — identifiers with no direct
     predictive signal for this exercise.
3. **Encoded categorical features**
   - `Sex` (binary) → **Label Encoding** (`male`=1, `female`=0).
   - `Embarked` (3 categories, nominal) → **One-Hot Encoding** with
     `drop_first=True` to avoid the dummy-variable trap.
4. **Feature scaling**
   - Engineered `FamilySize = SibSp + Parch + 1`.
   - Standardized `Age`, `Fare`, and `FamilySize` using `StandardScaler`
     (mean 0, std 1).
5. **Outlier detection & removal**
   - Visualized `Age` and `Fare` with boxplots before/after.
   - Removed outliers using the **IQR method** (1.5×IQR rule) — 127 rows
     were removed, leaving a cleaned dataset of 764 rows.

## What I Learned
Data cleaning, handling nulls (mean/median/mode imputation vs. feature
engineering for sparse columns), encoding categorical variables (label vs.
one-hot), feature scaling/standardization, and outlier detection/removal
using the IQR method.

---

## Interview Questions & Answers

**1. What are the different types of missing data?**
- **MCAR (Missing Completely at Random):** the missingness has no
  relationship to any observed or unobserved variable — purely random.
- **MAR (Missing at Random):** the missingness is related to other
  *observed* variables, but not to the missing value itself (e.g. older
  passengers being less likely to report age).
- **MNAR (Missing Not at Random):** the missingness is related to the
  *value itself* (e.g. people with very high incomes being less likely to
  disclose income).

**2. How do you handle categorical variables?**
Common approaches: **Label Encoding** (assigns each category an integer —
good for ordinal or binary categories), **One-Hot Encoding** (creates a
binary column per category — good for nominal categories with no order),
**Ordinal Encoding** (for categories with a genuine rank), and
**Target/Frequency Encoding** (replaces categories with a statistic derived
from the target or frequency — useful for high-cardinality features).

**3. What is the difference between normalization and standardization?**
- **Normalization (Min-Max Scaling)** rescales values into a fixed range,
  usually [0, 1]. Sensitive to outliers.
- **Standardization (Z-score Scaling)** rescales values to have mean 0 and
  standard deviation 1. It doesn't bound values to a fixed range but is
  less sensitive to outliers and is generally preferred for algorithms that
  assume normally distributed features (e.g. logistic regression, SVM,
  PCA).

**4. How do you detect outliers?**
- **Visually:** boxplots, scatter plots, histograms.
- **Statistically:** the **IQR method** (values beyond 1.5×IQR from Q1/Q3),
  **Z-score method** (values with |z| > 3), or domain-specific business
  rules.

**5. Why is preprocessing important in ML?**
Raw data is almost always incomplete, inconsistent, or in the wrong scale.
Models trained on messy data learn noise instead of signal, and many
algorithms fail entirely (or perform poorly) on missing values,
un-encoded text, or unscaled features. Preprocessing directly determines
how well the model can actually learn the real underlying pattern.

**6. What is one-hot encoding vs label encoding?**
**Label Encoding** assigns a single integer to each category (e.g.
red=0, green=1, blue=2) — this implies an ordinal relationship that may
not exist, which can mislead certain models. **One-Hot Encoding** creates
a separate binary column per category, avoiding a false sense of order,
but it increases dimensionality, especially for high-cardinality features.

**7. How do you handle data imbalance?**
- **Resampling:** oversampling the minority class (e.g. **SMOTE**) or
  undersampling the majority class.
- **Algorithmic:** using class weights (`class_weight='balanced'`) so the
  model penalizes misclassifying the minority class more heavily.
- **Better evaluation metrics:** precision, recall, F1-score, ROC-AUC
  instead of plain accuracy, which can be misleading on imbalanced data.
- **Ensemble methods** designed for imbalance, like **Balanced Random
  Forest** or **EasyEnsemble**.

**8. Can preprocessing affect model accuracy?**
Yes, significantly. Poor handling of missing values can bias the model or
throw errors; unscaled features can make distance-based or
gradient-based algorithms converge poorly or weight features incorrectly;
improper encoding can introduce false ordinal relationships; and leaving
outliers unchecked can skew the learned decision boundary. Good
preprocessing often has a bigger impact on final accuracy than the choice
of model itself.
