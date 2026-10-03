"""
generate_dataset.py
--------------------
This sandbox environment has no internet access, so the real Kaggle
Titanic CSV could not be downloaded directly. This script generates a
Titanic-STYLE dataset with the exact same columns, realistic value
distributions, missing values, and outliers as the real dataset, so the
full cleaning pipeline in task1_preprocessing.py works exactly the same
way it would on the original file.

If you have internet access, just download the real dataset from:
https://www.kaggle.com/competitions/titanic/data (train.csv)
and drop it into data/titanic.csv with the same column names — the rest
of the pipeline (task1_preprocessing.py) does not need to change at all.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 891  # same size as the real Kaggle Titanic training set

pclass = np.random.choice([1, 2, 3], size=N, p=[0.24, 0.21, 0.55])

# Sex correlated loosely with class just for realism
sex = np.random.choice(["male", "female"], size=N, p=[0.65, 0.35])

# Age depends a bit on class (older passengers more likely in 1st class)
age = np.where(
    pclass == 1,
    np.random.normal(38, 14, N),
    np.where(pclass == 2, np.random.normal(30, 13, N), np.random.normal(25, 12, N)),
)
age = np.clip(age, 0.42, 80)

sibsp = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.68, 0.23, 0.05, 0.02, 0.02])
parch = np.random.choice([0, 1, 2, 3], size=N, p=[0.76, 0.13, 0.08, 0.03])

fare = np.where(
    pclass == 1,
    np.random.gamma(4, 20, N),
    np.where(pclass == 2, np.random.gamma(3, 7, N), np.random.gamma(2, 5, N)),
)
# inject a handful of extreme outliers (like the real dataset has, e.g. Fare=512)
outlier_idx = np.random.choice(N, size=6, replace=False)
fare[outlier_idx] = np.random.uniform(250, 520, size=6)
fare = np.round(fare, 4)

embarked = np.random.choice(["S", "C", "Q"], size=N, p=[0.72, 0.19, 0.09])

# Survival roughly follows known Titanic patterns: women/children/1st class survive more
survive_prob = (
    0.25
    + 0.5 * (sex == "female")
    + 0.15 * (pclass == 1)
    + 0.05 * (pclass == 2)
    + 0.1 * (age < 12)
)
survive_prob = np.clip(survive_prob, 0.05, 0.95)
survived = np.random.binomial(1, survive_prob)

first_names_m = ["James", "John", "William", "Henry", "George", "Charles", "Frank", "Thomas"]
first_names_f = ["Mary", "Anna", "Margaret", "Elizabeth", "Alice", "Florence", "Emily", "Kate"]
last_names = ["Smith", "Brown", "Johnson", "Williams", "Jones", "Miller", "Davis", "Wilson",
              "Taylor", "Anderson", "Moore", "Clark", "Hill", "Baker", "Nelson"]

names = []
for s in sex:
    first = np.random.choice(first_names_m if s == "male" else first_names_f)
    last = np.random.choice(last_names)
    title = "Mr." if s == "male" else np.random.choice(["Mrs.", "Miss."])
    names.append(f"{last}, {title} {first}")

ticket = [f"{np.random.randint(10000, 99999)}" for _ in range(N)]

# Cabin: mostly missing (like the real dataset ~77% missing), only 1st/2nd class tend to have one
cabin = []
for p in pclass:
    if p == 1 and np.random.rand() < 0.6:
        cabin.append(f"{np.random.choice(list('ABCDEF'))}{np.random.randint(1, 150)}")
    elif p == 2 and np.random.rand() < 0.15:
        cabin.append(f"{np.random.choice(list('DEF'))}{np.random.randint(1, 150)}")
    else:
        cabin.append(np.nan)

df = pd.DataFrame({
    "PassengerId": np.arange(1, N + 1),
    "Survived": survived,
    "Pclass": pclass,
    "Name": names,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Ticket": ticket,
    "Fare": fare,
    "Cabin": cabin,
    "Embarked": embarked,
})

# Inject missing values into Age (~20%, matching the real dataset) and Embarked (2 rows)
age_missing_idx = np.random.choice(N, size=int(0.20 * N), replace=False)
df.loc[age_missing_idx, "Age"] = np.nan

embarked_missing_idx = np.random.choice(N, size=2, replace=False)
df.loc[embarked_missing_idx, "Embarked"] = np.nan

df.to_csv("data/titanic.csv", index=False)
print("Saved data/titanic.csv with shape:", df.shape)
print(df.isnull().sum())
