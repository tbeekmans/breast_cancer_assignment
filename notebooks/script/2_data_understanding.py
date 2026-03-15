#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Breast Cancer Classification
# 
# Author: Tobias Beekmans  
# Master ICT – Software Engineering  
# DataOps Specialization Project – Individual Assignment  
# Submission Date: 15.03.2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from scipy.stats import pointbiserialr
from breast_cancer_assignment.dataset import load_data

# Notebook settings
pd.set_option('display.max_columns', None)


# # 2.0 Data Understanding
# 
# According to the CRISP-DM methodology, the data understanding phase aims to explore the dataset in order to understand its structure, identify relevant patterns, and detect potential data quality issues. [1]
# 
# This phase includes collecting the data, describing its structure, verifying data quality, and exploring relationships between features and the target variable.

# ## 2.1 Data Collection
# 
# This section describes the source of the dataset and how it is accessed within the project workflow.
# 
# The dataset used in this project is the "Breast Cancer Wisconsin (Diagnostic)" dataset from the UCI Machine Learning Repository. [2]
# 
# The dataset contains 569 records [2], which is relatively small compared to many machine learning datasets.
# 
# Only a single data source is used in this project. No additional datasets are merged, which simplifies the data integration process and avoids potential inconsistencies between different sources.
# 
# Within the project structure, the dataset is managed in two stages:
# 
# - raw dataset stored in `data/raw`
# - processed datasets generated during the analysis in `data/processed`
# 
# Separating raw and processed data supports transparency and reproducibility of the analysis workflow.
# 
# To ensure consistent data access across notebooks, the dataset is loaded through the project function `load_data()`, which provides a standardized interface for retrieving the dataset.

df = load_data()


# ## 2.2 Dataset Description
# 
# This section describes the structure and characteristics of the dataset used in the analysis.  
# Following the CRISP-DM framework, the dataset is examined with respect to its size, variable types, and coding scheme.
# 
# The dataset consists of 569 observations and 30 numerical features describing morphological characteristics of cell nuclei extracted from digitized images of FNA samples of breast masses. [3]
# 
# Ten core characteristics of the cell nuclei are measured:
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
# For each characteristic, three different measurements are recorded:
# 
# - mean value
# - standard error
# - worst value
# 
# The dataset therefore contains 30 numerical predictor variables describing tumor cell morphology. In addition to these features, the dataset contains a diagnostic label indicating whether the tumor is malignant or benign. This variable serves as the target variable for the classification task. [2]
# 
# The following analysis examines the dataset structure by inspecting its dimensions, variable types, and sample records.

# ### Table Dimensions
# 
# The dimensions of the dataset are inspected to understand the number of records and variables.

rows, columns = df.shape
print(f"Rows: {rows}")
print(f"Columns: {columns}")


# The dataset contains 569 records and 31 columns in total.

# ### Data Types
# 
# The following output shows the data types of the variables in the dataset.

print(df.dtypes.value_counts())
df.dtypes


# All predictor variables are numerical. The dataset also contains one binary target variable representing the tumor diagnosis.
# 
# According to the dataset documentation [2], the encoding of the target variable is defined as:
# 
# - 0 = malignant tumor
# - 1 = benign tumor
# 
# This variable will be used as the prediction target for the machine learning models.

feature_df = df.select_dtypes(include='number').drop(columns=["target"])
feature_columns = feature_df.columns
n_samples = len(df)


# ### Sample Data
# 
# To obtain an initial understanding of the dataset, a small sample of observations is displayed.

df.sample(10)


# The sample shows that each observation contains numerical measurements describing cell properties together with the corresponding diagnostic label.

# ## 2.3 Data Quality Verification
# 
# The data quality verification step examines the dataset for potential issues that may affect later modeling stages. Typical data quality problems include missing values, duplicate records, invalid values, measurement errors or inconsistencies in the data representation.
# 
# Identifying such issues at an early stage helps ensure that the dataset is suitable for machine learning tasks and prevents modeling results from being affected by data errors or inconsistencies.

# ### Missing Values
# 
# The dataset is checked for missing values to verify that all observations contain complete information for each feature. Missing values may indicate data collection issues or incomplete measurements and can influence model performance if not handled properly.

missing_values = df.isna().sum()
missing_values_percent = df.isna().mean() * 100

missing_values_df = pd.DataFrame({
    "Missing Values [abs]": missing_values,
    "Missing Values [%]": missing_values_percent
})

print(f"Total missing values: {missing_values.sum()}")
missing_values_df.round(2)


# The inspection shows that the dataset does not contain missing values. All observations have complete feature information, meaning no imputation or removal of records is required.

# ### Duplicates
# 
# Duplicate observations are checked to ensure that the dataset does not contain repeated entries that could bias the statistical analysis or model training.

duplicates = df.duplicated().sum()
print(f"Total Duplicates: {duplicates}")


# The dataset does not contain duplicate records. Each observation therefore represents a unique tumor sample.

# ### Invalid Values
# 
# The dataset is examined for potential invalid or implausible values. In medical measurement data, certain values may be impossible or unlikely, such as negative measurements for physical properties like area, radius, or perimeter. Conducting plausibility checks helps detect potential measurement errors or incorrect data encoding.

negatives = (df.select_dtypes(include='number') < 0).any()
neg_columns = negatives[negatives].index.tolist()
print(f"Columns with negative values: {neg_columns if neg_columns else 'None'}")


# No invalid negative values were detected in the dataset. All feature values fall within reasonable numerical ranges and appear consistent with the expected measurement types.

# ### Outlier Detection
# 
# Outliers are examined to identify extreme observations that may influence statistical analysis or machine learning models.
# 
# Two common statistical approaches are used to identify potential outliers in numerical datasets [5]:
# 
# - **Interquartile Range (IQR):** The IQR method defines outliers as observations falling below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR (where IQR represents the interquartile range = Q3 − Q1)
# 
# - **Z-Score:** Observations with an absolute Z-Score greater than 3 are considered potential outliers (assumption of a normal distribution: approximately 99.7% of observations lie within three standard deviations of the mean)
# 
# Only numerical predictor variables are included in the outlier analysis. The target variable is excluded because it represents the diagnostic label.

# IQR Outlier
iqr_outliers = {}

for col in feature_columns:
    q1, q3 = feature_df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers[col] = ((feature_df[col] < lower) | (feature_df[col] > upper)).sum()
iqr_outliers = pd.Series(iqr_outliers)

iqr_percentage = (iqr_outliers / n_samples * 100).round(2)

# Z-Score Outlier
z_scores = feature_df.apply(zscore)
z_outliers = (abs(z_scores) > 3).sum()
z_percentage = (z_outliers / n_samples * 100).round(2)

# Outlier Dataframe
outlier_df = pd.DataFrame({
    "IQR Outliers": iqr_outliers,
    "IQR Outliers (%)": iqr_percentage,
    "Z-Score Outliers": z_outliers,
    "Z-Score Outliers (%)": z_percentage
})

print("Top 10 Features by IQR Outliers")
display(outlier_df.sort_values("IQR Outliers", ascending=False).head(10))

print("Top 10 Features by Z-Score Outliers")
display(outlier_df.sort_values("Z-Score Outliers", ascending=False).head(10))


# The outlier analysis shows that several variables contain observations outside typical statistical ranges. 
# 
# **IQR:**  
# The IQR method identifies the highest number of outliers in variables related to measurement variability. For example: the feature `area error` contains 65 observations identified as outliers using the IQR method, corresponding to approximately 11.4% of the dataset. Other variables (such as `radius error`, `perimeter error`, `worst area`) also reveal relatively high numbers of extreme values. 
# 
# Many of these outliers occur in variables representing measurement variability (features ending in "error") or the largest observed values of tumor characteristics (features labeled with "worst"). These variables represent measurement variability or extreme tumor characteristics and therefore tend to show larger deviations. [4]
# 
# **Z-Score:**  
# The Z-score method highlights a similar pattern. Variables such as `compactness error`, `symmetry error`, and `worst compactness` appear among the features with the highest number of extreme observations, indicating that several shape-related tumor characteristics also contain values that deviate strongly from the dataset mean.
# 
# In biomedical datasets such extreme observations do not necessarily represent data errors. They may reflect biological variability in tumor morphology or measurement variability in medical imaging data. [4]
# 
# Removing these observations without domain-specific justification may lead to the loss of relevant information. For this reason potential outliers are not removed at this stage. Their influence will instead be addressed during later preprocessing steps, for example through feature scaling or the use of machine learning models that are less sensitive to extreme values.

# ## 2.4 Data Exploration
# 
# This section explores patterns and relationships within the dataset in order to better understand the behavior of the features and their connection to the target variable.
# 
# Exploratory analysis helps to identify potentially relevant attributes, reveal hidden structures in the data, and formulate hypotheses about which variables may be most useful for the classification task.
# 
# The analysis focuses on descriptive statistics, class distribution, correlations between features, and relationships between features and the diagnostic outcome.

# ### 2.4.1 Statistical Overview
# 
# Descriptive statistics are calculated to obtain an overview of the numerical properties of the dataset.
# 
# Such statistics help to understand the scale, variability, and distribution of the features and provide an initial indication of potential issues such as skewed distributions or large differences in feature magnitude. These insights are important for later preprocessing steps such as feature scaling or transformation.

# ### Summary Statistics
# 
# Basic statistical measures are calculated to understand the range, central tendency, and spread of the numerical variables.

df.describe().round(2)


# The summary statistics provide an overview of the numerical properties of the dataset and reveal differences in feature scale and variability.
# 
# Several variables describing tumor size show larger numerical ranges than other measurements. Example: the feature `mean area` has an average value of approximately 655 with a maximum value above 2500, while features such as `mean smoothness` have values close to 0.1. Such differences in magnitude indicate that the dataset contains features with very different numerical scales.
# 
# In addition, some variables show high variability. For instance, the feature `area error` has a mean value of 40.34 with a standard deviation of 45.49 and a maximum value of 542.20, indicating variation across tumor samples.
# 
# Large differences between minimum and maximum values can also be observed in features such as `worst area`, which ranges from approximately 185 to over 4250. These wide ranges suggest strong variability in tumor characteristics within the dataset.
# 
# These observations indicate that feature scaling may be necessary before training certain machine learning algorithms, particularly models that rely on for example distance calculations.

# ### Additional Statistical Metrics
# 
# Additional statistical metrics are calculated to further analyze the distributional characteristics of the dataset features. These statistics help to better understand the properties of the features and support later modeling decisions, such as feature scaling or the selection of algorithms that are robust to non-normal data distributions.
# 
# Measures such as variance, skewness, and kurtosis provide information about variability, asymmetry, and the tail behavior of feature distributions. [5]

median = df.median(numeric_only=True)
variance = df.var(numeric_only=True)
skew = df.skew(numeric_only=True)
kurt = df.kurtosis(numeric_only=True)
iqr = df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True)

stats_df = pd.DataFrame({
    "Median": median,
    "Variance": variance,
    "Skewness": skew,
    "Kurtosis": kurt,
    "IQR": iqr
})

stats_df.round(2)


# **Skewness:**  
# Several variables exhibit noticeable positive skewness, indicating that their distributions contain a larger number of small values and a smaller number of extreme observations. This pattern is particularly visible in features such as `area error`, `concavity error`, and `perimeter error`, which show strong skewness values.
# 
# **Kurtosis:**  
# High kurtosis values can also be observed for several variables, suggesting the presence of heavy-tailed distributions. Such distributions contain more extreme observations than would be expected under a normal distribution, which is consistent with the outlier patterns observed earlier in the analysis.
# 
# **IQR:**  
# The interquartile range highlights differences in variability within the central 50% of the observations. Variables related to tumor size, such as `mean area` and `worst area`, exhibit substantially larger IQR values compared to other measurements, indicating higher variability in tumor size characteristics across samples.
# 
# These observations suggest that the dataset contains non-normally distributed variables and varying levels of dispersion, which may influence model behavior and should be considered during later preprocessing steps such as scaling or transformation.

# ### 2.4.2 Target Distribution
# 
# The distribution of the target variable is determined to assess whether the dataset is balanced between malignant and benign tumor cases.
# 
# Class imbalance can affect model performance and evaluation metrics.

df["target_label"] = df["target"].map({0: "Malignant", 1: "Benign"})

target_counts = df["target_label"].value_counts()
target_percent = (target_counts / len(df) * 100).round(1)

print(pd.DataFrame({"Count": target_counts, "Percentage": target_percent}))

plt.figure(figsize=(6,4))
sns.countplot(x="target_label", data=df)

plt.title("Class Distribution: Target Variable")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()


# The target distribution shows that benign tumor cases occur more frequently than malignant cases in the dataset. The dataset contains 357 benign cases (62.7%) and 212 malignant cases (37.3%).
# 
# Although the dataset is not perfectly balanced, the class distribution shows only a moderate imbalance and both classes are sufficiently represented for training classification models. Although the imbalance is moderate, class distribution remains relevant in medical classification tasks because performance can be affected when malignant cases are less frequent than benign cases. [6]
# 
# Class imbalance should be considered when evaluating model performance. In medical diagnosis tasks, correctly identifying malignant tumors is particularly important. Therefore, evaluation metrics such as recall, precision, and the confusion matrix will be considered in addition to overall accuracy.

# ### 2.4.3 Feature Correlation
# 
# A correlation matrix is calculated to examine relationships between the numerical features.
# 
# Highly correlated variables may capture similar information about tumor morphology. Such redundancy can lead to multicollinearity [5], which may influence certain machine learning algorithms, particularly linear models and distance-based methods. In this cases feature selection or regularization techniques may help reduce redundancy and improve model stability.

# #### Correlation Heatmap
# 
# The correlation heatmap visualises pairwise relationships between all numerical features. To improve readability, only the upper triangle of the correlation matrix is displayed, since the lower triangle would contain duplicate information.

corr_matrix = feature_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True, mask=mask, fmt='.2f', annot_kws={'size': 7}, cmap='vlag')

plt.title("Feature Correlation Matrix")
plt.show()


# The heatmap reveals several strong correlations between groups of variables.
# 
# Features describing tumor size show very high positive correlations. Measurements such as `mean radius`, `mean perimeter`, and `mean area` increase together because they represent related properties of the tumor cells.
# The corresponding “worst” measurements (`worst radius`, `worst perimeter`, `worst area`) also show strong correlations, indicating that extreme tumor characteristics follow similar geometric relationships.
# 
# These patterns suggest that several variables capture overlapping information about tumor morphology.

# #### Correlated Feature Pairs
# 
# To determine feature redundancy, pairs of variables with very high absolute correlation values are identified.

corr_matrix_abs = feature_df.corr().abs()

# Get upper triangle of the correlation matrix
upper_triangle = corr_matrix_abs.where(np.triu(np.ones_like(corr_matrix_abs), k=1).astype(bool))

high_corr_pairs = (upper_triangle.stack().sort_values(ascending=False).reset_index())
high_corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]

corr_threshold = 0.9
high_corr_pairs[high_corr_pairs["Correlation"] > corr_threshold].head(10)


# The analysis of highly correlated feature pairs confirms several strong relationships between variables (Correlation above 0.9).
# 
# In particular, measurements related to tumor size exhibit extremely high correlations. For example, `mean radius` and `mean perimeter` show a correlation of 0.998, while `mean radius` and `mean area` reach a correlation of approximately 0.987. Similar relationships can be observed for the corresponding “worst” measurements, such as `worst radius`, `worst perimeter`, and `worst area`.
# 
# These strong correlations are expected because these variables describe related geometric properties of tumor cell structures. For instance, larger cell radi naturally correspond to larger perimeters and areas.
# 
# Such strong correlations indicate a high degree of feature redundancy within the dataset. In machine learning models, this can lead to multicollinearity, which may affect model interpretability and stability. Therefore, feature selection or regularization techniques may be considered during later modeling stages to reduce redundancy.

# ### 2.4.4 Feature–Target Association
# 
# To examine how strongly individual features are associated with the diagnostic outcome, the relationship between each numerical feature and the target variable is analyzed.
# 
# Because the target variable represents a binary class label (malignant vs. benign), the point-biserial correlation coefficient is used. The point-biserial correlation measures the relationship between a continuous variable and a binary variable and can be interpreted as a special case of the Pearson correlation coefficient where one variable is dichotomous. [5]
# 
# Features with higher absolute point-biserial correlation values indicate stronger statistical association with the tumor diagnosis.

# Point-Biserial Correlation
pb_corr = {}

for col in feature_columns:
    corr, _ = pointbiserialr(feature_df[col], df["target"])
    pb_corr[col] = corr

corr_target = pd.Series(pb_corr).sort_values(key=lambda x: x.abs(), ascending=False)

# Plot top features
plt.figure(figsize=(10,5))
corr_target.head(10).plot(kind="bar")
plt.title("Top Features by Point-Biserial Correlation with Target")
plt.xlabel("Feature")
plt.ylabel("Correlation")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()


# The analysis reveals that several features exhibit strong statistical association with the tumor diagnosis.
# 
# The strongest correlations are observed for features describing tumor size and boundary irregularity. In particular, variables such as `worst concave points`, `worst perimeter`, `mean concave points`, and `worst radius` show the highest absolute correlations with the target variable.
# 
# All observed correlations are negative. This is expected because the target variable is encoded as 0 = malignant and 1 = benign. Larger values of these features therefore correspond to a higher likelihood of malignant tumors.
# 
# The results indicate that malignant tumors tend to exhibit larger cell nuclei and more irregular nuclear boundaries. Features related to concavity and concave points capture structural irregularities of the tumor cell border, while variables such as radius, perimeter, and area represent overall tumor size.
# 
# These findings are consistent with the dataset description and previous medical research on the Wisconsin Breast Cancer dataset, where geometric characteristics of cell nuclei were identified as important indicators for tumor malignancy. [3]
# 
# The results suggest that morphological measurements describing tumor size and structural irregularities provide strong predictive information for distinguishing malignant and benign tumor cases.

# ### 2.4.5 Feature Distribution Analysis
# 
# To further understand how features differ between malignant and benign tumors, the distributions of the most strongly correlated features were selected for visualization using boxplots and density plots.

top_features = corr_target.head(5).index.tolist()
n = len(top_features)

fig, axes = plt.subplots(n, 2, figsize=(20, 3 * n))
axes = np.atleast_2d(axes)

for i, feat in enumerate(top_features):

    sns.boxplot(data=df, x="target_label", y=feat, hue="target_label", palette='Set2', legend=False, ax=axes[i,0])

    axes[i,0].set_title(feat)
    axes[i,0].set_xlabel("Diagnosis")
    axes[i,0].set_ylabel("Value")

    sns.kdeplot(data=df, x=feat, hue="target_label", fill=True, alpha=0.3, ax=axes[i,1])

    axes[i,1].set_title(feat)
    axes[i,1].set_xlabel("Value")
    axes[i,1].set_ylabel("Density")

plt.tight_layout()
plt.show()


# The boxplots and density plots show clear distribution differences between malignant and benign tumor cases for the selected top features.
# 
# In particular, features such as `worst concave points`, `worst perimeter`, `worst radius`, `mean perimeter`, and `mean concave points` display noticeable separation between the two classes. For all five features, malignant tumors tend to have higher values than benign tumors.
# 
# The boxplots also indicate that malignant cases often show a wider spread and more extreme observations, suggesting greater variability within this class. The density plots confirm these findings by showing shifted distributions with only partial overlap between malignant and benign cases.
# 
# Overall, these visual patterns support the earlier correlation analysis and suggest that measurements related to tumor size and boundary irregularity provide meaningful information for distinguishing malignant from benign tumors.

# ## 2.5 Key Observations
# 
# The data understanding phase provides several important insights into the structure and characteristics of the dataset:
# 
# **1. High data quality**
# 
# The dataset is complete and well-structured. No missing values or duplicate records were detected, and all predictor variables are numerical. This simplifies the data preparation phase because no data imputation or record removal is required.
# 
# **2. Differences in feature scale and distribution**
# 
# The descriptive statistics show that the variables differ in scale and distribution. Features related to tumor size, such as `area`, `perimeter`, and `radius`, have much larger numerical ranges than other variables. This indicates that feature scaling may be required for certain machine learning algorithms.
# 
# **3. Strong feature correlations**
# 
# The correlation analysis revealed several highly correlated feature pairs, particularly among variables describing tumor size and related geometric properties. This indicates a degree of feature redundancy within the dataset and suggests that feature selection or regularization techniques may be useful during model development.
# 
# **4. Strong association between morphological features and tumor diagnosis**
# 
# The point-biserial correlation analysis showed that several features are strongly associated with the target variable. In particular, measurements related to tumor size and boundary irregularity (such as radius, perimeter, area, and concave points) appear to be highly informative for distinguishing malignant from benign tumors.
# 
# **5. Clear distribution differences between tumor classes**
# 
# The feature distribution analysis demonstrated noticeable differences between malignant and benign tumor cases. Malignant tumours tend to exhibit larger values and greater variability for several morphological measurements. These patterns confirm that the extracted features contain meaningful diagnostic information and are suitable for machine learning classification tasks.

# ## References
# 
# [1] IBM Corporation (2011): *IBM SPSS Modeler CRISP-DM Guide*
# 
# [2] UCI Machine Learning Repository (1995): *Breast Cancer Wisconsin (Diagnostic)*. Retrieved from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# [3] Street, W. N.; Wolberg, W. H.; Mangasarian, O. L. (1993): *Nuclear feature extraction for breast tumor diagnosis*
# 
# [4] Sidey-Gibbons, J. A. M.; Sidey-Gibbons, C. J. (2019): *Machine learning in medicine: a practical introduction*
# 
# [5] James, G.; Witten, D.; Hastie, T.; Tibshirani, R. (2023): *An Introduction to Statistical Learning with Applications in Python*
# 
# [6] Basu, U.; Yan, Y. (2025): *Data Balancing in Breast Cancer Recognition Using the Wisconsin Breast Cancer Dataset*
