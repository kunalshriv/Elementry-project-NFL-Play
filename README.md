import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('NFL Play by Play 2009-2016 (v3).csv')

# Display the first few rows of the dataset
print(df.head())

# Display summary statistics
print(df.describe())

# Display information about the dataset
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

# Impute missing values (e.g., using mean for numerical columns)
df.fillna(df.mean(), inplace=True)

# Alternatively, drop rows/columns with missing values
# df.dropna(inplace=True)

# Using IQR to identify outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Using Z-score to identify outliers
from scipy import stats
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


# Standardize date formats
df['date_column'] = pd.to_datetime(df['date_column'])

# Standardize categorical values
df['category_column'] = df['category_column'].str.lower().str.strip()

# Correct text data errors
df['text_column'] = df['text_column'].str.replace('incorrect_value', 'correct_value')


# Remove duplicate rows
df.drop_duplicates(inplace=True)


# Normalize/scale data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Alternatively, use MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


# Visualize distributions
df.hist(bins=30, figsize=(20, 15))
plt.show()

# Visualize correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Visualize relationships
sns.pairplot(df)


# Check for any remaining issues
print(df.isnull().sum())
print(df.info())
print(df.describe())


