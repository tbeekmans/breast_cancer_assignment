#!/usr/bin/env python
# coding: utf-8

# # 2.0 Data Understanding
# 
# ## Goal
# Understand the dataset structure, basic quality, and the target distribution.
# 
# ## Data Sources
# - Local Raw Data: `data/raw/dataset.csv` (gitignored)
# - Processed Copy: `data/processed/dataset.csv` (gitignored, created by `make data`)

# Importing Libraries
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Configuring Notebook Settings
pd.set_option('display.max_columns', None)

# Loading Data
df = pd.read_csv("../data/raw/dataset.csv")


# ### Sample Data

display(df.sample(10))


# ### Table Dimensions

rows = df.shape[0]
columns= df.shape[1]
print("\nRows:" + str(rows))
print("Columns:" + str(columns))


# ### Data Types

print(df.dtypes.value_counts())
display(df.dtypes)


# ### Data Quality

missing_values = df.isna().sum()
missing_percent = df.isna().mean() * 100
duplicates = df.duplicated().sum()
unnamed_columns = [col for col in df.columns if "unnamed" in col.lower()]
negatives = (df.select_dtypes(include='number') < 0).any()
neg_columns = negatives[negatives].index.tolist()

data_quality = pd.DataFrame({
    'Missing Values': missing_values,
    'Missing [%]': missing_percent
})
display(data_quality.round(2))

print(f"Total Duplicates: {duplicates}")
print(f"Unnamed Columns: {unnamed_columns if unnamed_columns else 'None'}")
print(f"Columns with negative values: {neg_columns if neg_columns else 'None'}")
print(f"Columns with missing values: {(missing_values > 0).sum()}")
print(f"Total missing values: {missing_values.sum()}")


# ### Outlier-Check

# IQR Outlier Count
numeric_columns = df.select_dtypes(include='number').columns
iqr_outliers = {}
for col in numeric_columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()
iqr_outliers = pd.Series(iqr_outliers)

# Z-Score Outlier Count
z_scores = df[numeric_columns].apply(zscore)
z_outliers = (abs(z_scores) > 3).sum()

# Compare
outlier_compare = pd.DataFrame({
    'Z-Score Outliers': z_outliers,
    'IQR Outliers': iqr_outliers
})
display(outlier_compare.sort_values('IQR Outliers', ascending=False).head(10))


# ### Summary Statistics

display(df.describe().round(2))


# ### Additional Metrics

median = df.median(numeric_only=True)
mode = df.mode(numeric_only=True).iloc[0]
variance = df.var(numeric_only=True)
skew = df.skew(numeric_only=True)
kurt = df.kurtosis(numeric_only=True)
iqr = df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True)

data_stats = pd.DataFrame({
    'median': median,
    'mode': mode,
    'variance': variance,
    'skewness': skew,
    'kurtosis': kurt,
    'IQR': iqr
}).T

display(data_stats.round(2))


# ## Visualization

# ### Target Variable
# 
# 

df['target_label'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

plt.figure(figsize=(6,4))
ax = sns.countplot(x='target_label', data=df)
plt.title('Distribution: Target Variable')
plt.xlabel('Diagnosis')
plt.ylabel('Count')

display(df['target'].value_counts())
plt.show()


# ### Correlation Heatmap

df_numeric = df.drop(columns=['target', 'target_label'])
plt.figure(figsize=(24,16))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f', cmap='vlag')
plt.title('Correlation Matrix Heatmap')
plt.show()


# ### Feature Correlation with Target

feature_columns = df.select_dtypes(include='number').columns.drop('target')

# Correlation of each numeric feature with the numeric target
corr_target = df[feature_columns].corrwith(df['target']).sort_values(key=lambda x: x.abs(), ascending=False)

print("Top 5 positive correlated features:")
display(corr_target[corr_target > 0].sort_values(ascending=False).head(5))

print("Top 5 negative correlated features:")
display(corr_target[corr_target < 0].sort_values(ascending=True).head(5))

# Plot top correlations by absolute magnitude
plt.figure(figsize=(10, 5))
corr_target.head(10).plot(kind='bar')
plt.title('Top 10 Features by Correlation with Target')
plt.xlabel('Feature')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ### Feature Analysis

top_features = corr_target.head(10).index.tolist()
n = len(top_features)

fig, axes = plt.subplots(n, 2, figsize=(14, 2.4 * n))
axes = np.atleast_2d(axes)

for i, feat in enumerate(top_features):
    # LEFT: boxplot
    sns.boxplot(
        data=df,
        x="target_label",
        y=feat,
        hue="target_label",
        palette="Set2",
        dodge=False,
        ax=axes[i, 0],
    )
    axes[i, 0].set_title(feat)
    axes[i, 0].set_xlabel("target_label")
    axes[i, 0].set_ylabel(feat)

    # RIGHT: kde
    sns.kdeplot(
        data=df,
        x=feat,
        hue="target_label",
        fill=True,
        common_norm=False,
        alpha=0.30,
        ax=axes[i, 1],
    )
    axes[i, 1].set_title(feat)
    axes[i, 1].set_xlabel(feat)
    axes[i, 1].set_ylabel("density")

plt.tight_layout()
plt.show()

