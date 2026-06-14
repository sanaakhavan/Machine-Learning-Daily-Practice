# Day 1: Handling Missing Data in Python
# ---------------------------------------------------------------------------------
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# 1. Loading the dataset
dataset = pd.read_csv("smartphone.csv")

# Separating features (X) and target variable (y)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 2.For numerical data: replace missing values with the column's mean
impute_num = SimpleImputer(missing_values=np.nan, strategy="mean")

# 3. For categorical data: replace missing values with the most frequent category
impute_country = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# 4. Applying Imputers to the data
x[:, 1:3] = impute_num.fit_transform(x[:, 1:3])

x[:, 0:1] = impute_country.fit_transform(x[:, 0:1])

y = impute_country.fit_transform(y.reshape(-1, 1))

print("X =")
print(x)

print(" Y =")
print(y)
