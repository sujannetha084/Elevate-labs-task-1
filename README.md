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


