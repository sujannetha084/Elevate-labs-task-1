"""
Task 1: Data Cleaning & Preprocessing
AI & ML Internship - Elevate Labs

Objective: Learn how to clean and prepare raw data for ML.
Dataset: Titanic dataset (data/titanic.csv)

Steps covered:
1. Import the dataset and explore basic info (nulls, data types)
2. Handle missing values using mean/median/imputation
3. Convert categorical features into numerical using encoding
4. Normalize/standardize the numerical features
5. Visualize outliers using boxplots and remove them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set_style("whitegrid")

# ---------------------------------------------------------------
# STEP 1: Import the dataset and explore basic info
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOAD DATA & EXPLORE")
print("=" * 60)

df = pd.read_csv("data/titanic.csv")

print("\nShape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values Heatmap (Before Cleaning)")
plt.tight_layout()
plt.savefig("images/missing_values_heatmap.png", dpi=120)
plt.close()
print("\nSaved images/missing_values_heatmap.png")

# ---------------------------------------------------------------
# STEP 2: Handle missing values
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: HANDLE MISSING VALUES")
print("=" * 60)

# Age -> numeric, right-skewed a bit -> use median imputation (robust to skew/outliers)
df["Age"] = df["Age"].fillna(df["Age"].median())
print("\nFilled missing 'Age' with median:", df["Age"].median())

# Embarked -> categorical with very few missing -> use mode imputation
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("Filled missing 'Embarked' with mode:", df["Embarked"].mode()[0])

# Cabin -> ~80% missing, too sparse to impute meaningfully.
# Instead of dropping all information, engineer a binary feature and then drop the column.
df["Has_Cabin"] = df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin"], inplace=True)
print("Engineered 'Has_Cabin' binary feature and dropped 'Cabin' column (too sparse to impute).")

# Drop columns that are pure identifiers / free text with no direct predictive value
# for this preprocessing exercise (Name, Ticket, PassengerId).
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# ---------------------------------------------------------------
# STEP 3: Encode categorical features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: ENCODE CATEGORICAL FEATURES")
print("=" * 60)

# Sex -> binary categorical -> Label Encoding (male=1, female=0)
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print("\nLabel encoded 'Sex':", dict(zip(le.classes_, le.transform(le.classes_))))

# Embarked -> nominal categorical with 3 categories -> One-Hot Encoding
df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", drop_first=True)
print("One-hot encoded 'Embarked' ->", [c for c in df.columns if c.startswith("Embarked")])

print("\nColumns after encoding:\n", df.columns.tolist())

# ---------------------------------------------------------------
# STEP 4: Feature engineering + Normalize/standardize numerical features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: FEATURE SCALING")
print("=" * 60)

# Small feature engineering step: combine SibSp + Parch into FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

numeric_cols = ["Age", "Fare", "FamilySize"]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"\nStandardized columns {numeric_cols} to mean=0, std=1 using StandardScaler.")
print(df[numeric_cols].describe())

# ---------------------------------------------------------------
# STEP 5: Visualize and remove outliers
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: OUTLIER DETECTION & REMOVAL")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=df["Age"], ax=axes[0], color="skyblue")
axes[0].set_title("Age (scaled) - Before Outlier Removal")
sns.boxplot(y=df["Fare"], ax=axes[1], color="salmon")
axes[1].set_title("Fare (scaled) - Before Outlier Removal")
plt.tight_layout()
plt.savefig("images/boxplots_before_outlier_removal.png", dpi=120)
plt.close()
print("\nSaved images/boxplots_before_outlier_removal.png")


def remove_outliers_iqr(data, column):
    """Remove rows where `column` falls outside 1.5*IQR from Q1/Q3."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[column] >= lower) & (data[column] <= upper)]


rows_before = df.shape[0]
df = remove_outliers_iqr(df, "Fare")
df = remove_outliers_iqr(df, "Age")
rows_after = df.shape[0]
print(f"\nRemoved {rows_before - rows_after} outlier rows using the IQR method "
      f"(kept {rows_after}/{rows_before} rows).")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=df["Age"], ax=axes[0], color="skyblue")
axes[0].set_title("Age (scaled) - After Outlier Removal")
sns.boxplot(y=df["Fare"], ax=axes[1], color="salmon")
axes[1].set_title("Fare (scaled) - After Outlier Removal")
plt.tight_layout()
plt.savefig("images/boxplots_after_outlier_removal.png", dpi=120)
plt.close()
print("Saved images/boxplots_after_outlier_removal.png")

# ---------------------------------------------------------------
# Save the final cleaned dataset
# ---------------------------------------------------------------
df.to_csv("data/titanic_cleaned.csv", index=False)
print("\n" + "=" * 60)
print(f"DONE. Final cleaned dataset shape: {df.shape}")
print("Saved to data/titanic_cleaned.csv")
print("=" * 60)
