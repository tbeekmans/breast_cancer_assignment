#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Classify Breast Cancer Cases
# 
# Author: Tobias Beekmans  
# Master ICT – Software Engineering  
# DataOps Specialisation Project – Individual Assignment  
# Submission Date: 15.03.2026
# 
# **Short Description:**
# The Breast Cancer dataset [1] is a widely used dataset for learning and practicing machine learning techniques. It contains diagnostic data for breast cancer cases, including features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei, such as radius, texture, and smoothness, and are compiled into a convenient dataset.
# 
# **Goal:**
# Develop a machine learning model to accurately classify breast cancer cases as malignant or benign.

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import pointbiserialr
from breast_cancer_assignment.dataset import load_data

# Configuring Notebook Settings
pd.set_option('display.max_columns', None)


# # 2.0 Data Understanding

# ## 2.1 Data Sources and Collection
# 
# The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository [1]. It contains diagnostic measurements derived from digitized images of fine needle aspirate (FNA) samples of breast masses.
# 
# Within this project, the data is managed in two local forms:
# 
# - original raw dataset: data/raw/dataset.csv
# - processed dataset generated through the project workflow: data/processed/dataset.csv
# 
# To ensure reproducibility and consistency across notebooks, the dataset is loaded through the project function `load_data()`, which provides a standardised access point for the data used in the analysis.

df = load_data()


# ## 2.2 Dataset Description
# 
# The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository [1].
# 
# The dataset contains features extracted from digitized images of fine needle aspirate (FNA) samples of breast masses. These features describe morphological characteristics of the cell nuclei present in the images and are commonly used for diagnostic classification tasks. The features represent ten core characteristics of the cell nuclei [1]:
# 
# - radius
# - texture
# - perimeter
# - area
# - smoothness
# - compactness
# - concavity
# - concave points
# - symmetry
# - fractal dimension
# 
# For each of these characteristics, three measurements are provided:
# 
# - mean value
# - standard error
# - worst value
# 
# These measurements capture both the typical properties of the nuclei and variations within the tumour sample.
# 
# The dataset also contains a diagnostic label indicating whether the tumour is malignant or benign, which serves as the target variable for the classification task.
# 
# The following analysis inspects the dataset structure by examining its dimensions, column data types, and sample data.

# ### Table Dimensions
# 
# The dimensions of the dataset are inspected to understand the number of records and variables.

rows, columns = df.shape
print(f"Rows: {rows}")
print(f"Columns: {columns}")


# The dataset contains 569 records and 31 columns in total.

# ### Data Types
# 
# The following output shows the data type of each column.

print(df.dtypes.value_counts())
df.dtypes


# All variables are numerical. The dataset also contains one target variable that represents the class label used for classification. According to the dataset documentation [1], the encoding is defined as:
# 
# - 0 = malignant tumour
# - 1 = benign tumour
# 
# This variable will later be used as the prediction target for the machine learning models.

# ### Sample Data
# 
# To gain an initial understanding of the dataset, a small sample of observations is displayed.

df.sample(10)


# The sample shows that each observation contains numerical measurements describing cell properties, together with a target variable indicating whether the tumour is malignant or benign.

# ## 2.3 Data Structure Overview
# 
# This section examines the statistical properties of the dataset features.  
# Descriptive statistics help to understand the scale, distribution, and variability of the numerical variables before further analysis and modelling steps.
# 
# The first overview is generated using the standard `describe()` function in pandas, which provides key statistics for each numerical feature, including the mean, standard deviation, minimum, maximum, and quartile values.

# ### Summary Statistics
# 
# Basic statistical measures are calculated to understand the range, central tendency, and spread of the numerical variables.

df.describe().round(2)


# The summary statistics show that the dataset contains numerical variables with varying scales and ranges.
# 
# Some features such as `area_mean` and `perimeter_mean` have much larger values compared to variables such as `smoothness_mean` or `fractal_dimension_mean`.  
# This indicates that feature scaling may be necessary before training certain machine learning models.
# 
# In addition, the presence of large differences between minimum and maximum values suggests that some features may contain potential outliers, which will be investigated further in the data quality analysis.

# ### Additional Statistical Metrics
# 
# Additional statistical metrics are calculated to further examine the distribution and variability of the numerical features.
# 
# These metrics provide complementary information to the summary statistics and help identify potential asymmetries or dispersion patterns within the dataset.

median = df.median(numeric_only=True)
variance = df.var(numeric_only=True)
skew = df.skew(numeric_only=True)
kurt = df.kurtosis(numeric_only=True)
iqr = df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True)

stats = pd.DataFrame({
    'median': median,
    'variance': variance,
    'skewness': skew,
    'kurtosis': kurt,
    'IQR': iqr
})

stats.round(2)


# The additional statistical metrics provide further insight into the distribution characteristics of the dataset.
# 
# Several variables exhibit noticeable skewness, indicating that their distributions are not perfectly symmetric.  
# This suggests that some features may contain long-tailed distributions or extreme values.
# 
# The interquartile range (IQR) highlights differences in variability between features, which may influence model behaviour and should be considered during later preprocessing steps such as scaling or feature transformation.

# ## 2.4 Data Quality Verification
# 
# Before proceeding to further analysis and modelling, the dataset is examined for potential data quality issues.
# 
# Typical data quality problems include missing values, duplicate records, invalid values, and extreme outliers. Identifying such issues early is important to ensure that the dataset is suitable for machine learning tasks and that model results are not distorted by data errors.

# ### Missing Values
# 
# The dataset is checked for missing values to ensure that all observations contain complete feature information.

missing_values = df.isna().sum()
missing_values_percent = df.isna().mean() * 100

missing_values_df = pd.DataFrame({
    'Missing Values [abs]': missing_values,
    'Missing Values [%]': missing_values_percent
})

print(f"Total missing values: {missing_values.sum()}")
missing_values_df.round(2)


# The inspection shows that the dataset does not contain missing values.  
# All observations have complete feature information, meaning no imputation or removal of records is required.

# ### Duplicates
# 
# Duplicate observations are checked to ensure that the dataset does not contain repeated entries that could bias the model training.

duplicates = df.duplicated().sum()
print(f"Total Duplicates: {duplicates}")


# The dataset does not contain duplicate records.  
# Each observation therefore represents a unique sample.

# ### Invalid Values
# 
# The dataset is examined for potential invalid values, such as negative values in features where negative measurements would not be meaningful.

negatives = (df.select_dtypes(include='number') < 0).any()
neg_columns = negatives[negatives].index.tolist()
print(f"Columns with negative values: {neg_columns if neg_columns else 'None'}")


# No invalid negative values were detected in the dataset.  
# All feature values appear to fall within reasonable numerical ranges.

# ### Outlier Detection
# 
# Outliers are examined to identify extreme values that may influence model training.
# 
# Two common statistical methods are used to identify potential extreme values:
# 
# - Interquartile Range (IQR): identifies values outside the typical distribution range
# - Z-score: detects observations that are several standard deviations away from the mean
# 
# These methods help identify features that may contain extreme observations. However, in medical datasets such values may reflect genuine biological variation rather than data errors.

numeric_columns = df.select_dtypes(include='number').columns.drop('target')

# IQR Outlier Count
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

outlier_compare.sort_values('IQR Outliers', ascending=False).head(10)


# The outlier analysis indicates that several features contain observations that fall outside typical statistical ranges.
# 
# Such values may reflect natural biological variability in tumour cell structures rather than measurement errors. Therefore, outliers will not be removed at this stage but will be considered during later preprocessing steps.
# 
# If necessary, robust scaling techniques or model types that are less sensitive to extreme values may be used.

# ## 2.5 Data Exploration
# 
# This section explores relationships and patterns within the dataset in order to better understand the behaviour of the features and their connection to the target variable.
# 
# The analysis focuses on the distribution of the target variable, correlations between features, and how individual features differ between malignant and benign tumour cases.

# ### 2.5.1 Target Distribution
# 
# The distribution of the target variable is examined to determine whether the dataset is balanced between malignant and benign tumour cases.
# 
# Class imbalance can affect model performance and evaluation metrics.

df['target_label'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

target_counts = df['target_label'].value_counts()
print(target_counts)

plt.figure(figsize=(6,4))
sns.countplot(x='target_label', data=df)
plt.title('Distribution: Target Variable')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()


# The distribution shows that benign cases occur more frequently than malignant cases in the dataset. Although the classes are not perfectly balanced, the imbalance is moderate and both classes are sufficiently represented.
# 
# This is important for model training, as extremely imbalanced datasets can lead to biased models that favour the majority class. In this dataset, the class distribution is still suitable for standard classification techniques, though evaluation metrics such as recall and precision remain important.

# ### 2.5.2 Feature Correlation
# 
# A correlation matrix is calculated to examine relationships between the numerical features.
# 
# Highly correlated variables may capture similar information about tumour morphology. Such redundancy can influence certain machine learning algorithms, particularly linear models and distance-based methods, and may later motivate feature selection or regularisation.

# #### Correlation Heatmap
# 
# The correlation heatmap visualises pairwise relationships between all numerical features.

df_numeric = df.select_dtypes(include="number").drop(columns=["target"])

plt.figure(figsize=(20,14))
sns.heatmap(df_numeric.corr(), annot=True, fmt='.2f', cmap='vlag', annot_kws={"size": 7})
plt.title('Feature Correlation Matrix')
plt.show()


# The heatmap reveals several strong correlations between groups of variables.
# 
# This is expected because multiple features describe related geometric properties of cell nuclei. For example, measurements such as radius, perimeter, and area naturally increase together and therefore show strong positive correlations.
# 
# The presence of strong correlations suggests that some features may contain overlapping information.

# #### Highly Correlated Feature Pairs
# 
# To further examine feature redundancy, pairs of variables with very high absolute correlation values are identified.

corr_matrix = df_numeric.corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_pairs = (
    upper_triangle.stack()
    .sort_values(ascending=False)
    .reset_index()
)

high_corr_pairs.columns = ["Feature 1", "Feature 2", "Absolute Correlation"]

high_corr_pairs[high_corr_pairs["Absolute Correlation"] > 0.90].head(10)


# The results confirm that several feature pairs exhibit very high correlation values.
# 
# This indicates that the dataset contains substantial feature redundancy. Such redundancy is expected because the dataset includes multiple measurements (mean, standard error, and worst values) for related morphological characteristics of the tumour cells.
# 
# These observations will be considered during later modelling stages, where feature selection or regularisation techniques may help reduce redundancy and improve model stability.

# ### 2.5.3 Feature–Target Association
# 
# To examine how strongly individual features are associated with the diagnostic outcome, the relationship between each numerical feature and the target variable is analysed.
# 
# Because the target variable represents a binary class label (malignant vs. benign), a standard Pearson correlation is not strictly appropriate. Instead, the point-biserial correlation coefficient is used. This statistic measures the relationship between a continuous variable and a binary variable and can therefore be interpreted as a special case of Pearson correlation adapted for binary classification problems.
# 
# Features with higher absolute point-biserial correlation values indicate stronger statistical association with the tumour diagnosis.

feature_columns = df.select_dtypes(include='number').columns.drop('target')

# Correlation
pb_corr = {}

for col in feature_columns:
    corr, _ = pointbiserialr(df[col], df['target'])
    pb_corr[col] = corr

corr_target = pd.Series(pb_corr).sort_values(key=lambda x: x.abs(), ascending=False)

# Plot
plt.figure(figsize=(10, 5))
corr_target.head(10).plot(kind='bar')
plt.title('Top Features by Point-Biserial Correlation with Target')
plt.xlabel('Feature')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# The analysis shows that several features exhibit strong statistical association with the tumour diagnosis.
# 
# In particular, measurements related to tumour size and shape, such as radius, perimeter, and area, appear to have strong relationships with the target variable. This suggests that these features may be highly informative for distinguishing malignant and benign tumour cases.
# 
# These results support the suitability of the dataset for machine learning classification tasks and help identify variables that may contribute significantly to model performance.

# ### 2.5.4 Feature Distribution Analysis
# 
# To further understand how features differ between malignant and benign tumours, the distributions of the most strongly correlated features were selected for visualization using boxplots and density plots.

top_features = corr_target.head(10).index.tolist()
n = len(top_features)

fig, axes = plt.subplots(n, 2, figsize=(14, 2.4 * n))
axes = np.atleast_2d(axes)

for i, feat in enumerate(top_features):
    sns.boxplot(
        data=df,
        x="target_label",
        y=feat,
        hue="target_label",
        palette="Set2",
        legend=False,
        ax=axes[i, 0],
    )
    axes[i, 0].set_title(feat)
    axes[i, 0].set_xlabel("Diagnosis")
    axes[i, 0].set_ylabel(feat)

    sns.kdeplot(
        data=df,
        x=feat,
        hue="target_label",
        fill=True,
        alpha=0.3,
        ax=axes[i, 1],
    )
    axes[i, 1].set_title(feat)
    axes[i, 1].set_xlabel(feat)
    axes[i, 1].set_ylabel("Density")

plt.tight_layout()
plt.show()


# The visualisations show that several features exhibit clear differences between malignant and benign tumour cases.
# 
# In many cases, malignant tumours tend to have larger values for size-related measurements such as radius, perimeter, and area. The distributions also appear to be more spread out for malignant cases, indicating greater variability.
# 
# These differences suggest that the extracted morphological features contain meaningful information for distinguishing between tumour types, supporting the suitability of the dataset for classification tasks.

# ## 2.6 Key Observations
# 
# The data understanding phase provides several important insights into the structure and characteristics of the dataset.
# 
# First, the dataset is complete and well-structured. No missing values or duplicate records were detected, and all predictor variables are numerical. This simplifies the preprocessing stage because no data imputation or record removal is required.
# 
# Second, the statistical analysis shows that the features vary considerably in scale and distribution. Some variables, such as measurements related to tumour size, have much larger numerical ranges than others. This indicates that feature scaling may be necessary for certain machine learning algorithms.
# 
# Third, the exploratory analysis reveals that several features are strongly correlated with each other. This confirms that the dataset contains a degree of feature redundancy, which is expected because multiple measurements describe related morphological characteristics of the tumour cells.
# 
# Finally, several features related to tumour size and shape exhibit strong association with the diagnostic label. This suggests that the extracted morphological measurements contain meaningful information for distinguishing malignant from benign tumour cases.
# 
# Overall, the dataset appears suitable for machine learning classification tasks. However, the next phase of the CRISP-DM process will need to address feature scaling and consider the potential impact of correlated variables during model development.
