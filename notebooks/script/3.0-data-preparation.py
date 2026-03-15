#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Breast Cancer Classification
# 
# Author: Tobias Beekmans  
# Master ICT – Software Engineering  
# DataOps Specialisation Project – Individual Assignment  
# Submission Date: 15.03.2026

# Import libraries
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from breast_cancer_assignment.dataset import load_data

# Notebook settings
pd.set_option("display.max_columns", None)

# Load dataset
df = load_data()


# # 3.0 Data Preparation
# 
# According to the CRISP-DM methodology, the data preparation phase focuses on transforming the dataset into a form suitable for modeling. [1]
# 
# Based on the findings from the data understanding phase, preparation in this project focuses on organizing the predictor and target variables, splitting the dataset into training and test subsets, and applying feature scaling where required for the selected machine learning algorithms.

# ## 3.1 Data Selection
# 
# All predictor variables are retained for modeling. Although the data understanding phase revealed correlations between several variables, the full predictor set is preserved in order to retain the complete diagnostic information available in the dataset.
# 
# The variable `target` represents the tumour diagnosis and is used as the response variable for the classification task.

# Define feature matrix and target variable
X = df.drop(columns="target")
y = df["target"]

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# The resulting feature matrix contains 569 observations and 30 predictor variables, while the target vector contains 569 binary diagnostic labels.

print("Selected feature columns:")
print(X.columns.tolist())


# The selected predictors include all diagnostic measurements provided in the dataset. No attributes are excluded, ensuring that the complete set of morphological features is available for model training.

# ## 3.2 Data Cleaning
# 
# The data understanding phase included a comprehensive data quality assessment. This assessment showed that the dataset does not contain missing values, duplicate observations, or invalid numerical measurements. Therefore, no imputation or record removal is required before modeling.
# 
# Potential outliers were identified during exploratory analysis. These observations are retained because extreme values in biomedical datasets may represent genuine biological variation rather than measurement errors. [4]
# 
# Consequently, no observations or attributes are removed during the data cleaning phase.

# ## 3.3 Data Construction
# 
# No additional attributes or records are constructed. The dataset already consists of domain-specific diagnostic measurements extracted from fine needle aspirate (FNA) images of breast masses, which were designed to support tumour classification tasks. [2]
# 
# Because these variables already encode relevant geometric and structural characteristics of tumour cells, further feature construction is not required for the modeling workflow.

# ## 3.4 Train–Test Split
# 
# To evaluate machine learning models, the dataset must be divided into separate training and test subsets. The training data is used to fit the machine learning models, while the test data is reserved for evaluating final model performance on previously unseen observations. This separation helps prevent overly optimistic performance estimates and allows a more realistic assessment of how well a model generalizes to new data. [3]
# 
# In classification problems it is important to preserve the original class distribution when splitting the dataset. If the proportion of malignant and benign tumours changes significantly between training and test sets, the evaluation results may become biased. For this reason, a stratified sampling strategy is used. Stratified splitting ensures that both subsets maintain approximately the same class proportions as the original dataset. [5]
# 
# The dataset is divided using an 80/20 split, where 80% of the data is used for model training and 20% is reserved for testing. This ratio is commonly used in machine learning projects because it provides sufficient training data while keeping an independent subset for evaluation. [3]
# 
# A fixed random seed (random_state = 42) is used to ensure that the data split is reproducible.

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training feature matrix:", X_train.shape)
print("Test feature matrix:", X_test.shape)

print("Training target vector:", y_train.shape)
print("Test target vector:", y_test.shape)


# The output confirms that 455 observations are assigned to the training set and 114 observations to the test set, corresponding to an 80/20 split.

print("Training class distribution:")
print(y_train.value_counts(normalize=True))

print("Test class distribution:")
print(y_test.value_counts(normalize=True))


# The class distributions of benign and malignant tumours in the training and test sets are nearly identical, confirming that stratified sampling preserved the original class balance of the dataset.

# ## 3.5 Feature Scaling
# 
# The predictor variables are standardized using "StandardScaler". Standardization transforms each feature to have a mean of approximately 0 and a standard deviation of approximately 1, ensuring that all variables operate on a comparable numerical scale. [3]
# 
# The scaler is fitted only on the training data and then applied to both the training and test sets. This approach prevents data leakage and ensures that information from the test data does not influence the training process. [5]

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_scaled = scaler.transform(X_test)

print("Scaled training set shape:", X_train_scaled.shape)
print("Scaled test set shape:", X_test_scaled.shape)

scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

print(scaled_df.describe().round(2))


# The descriptive statistics confirm that the scaling procedure was applied correctly. Across all features, the mean values are approximately 0 and the standard deviations are approximately 1, which is the expected result of standardization.
# 
# The transformation does not change the number of observations or features, it only adjusts the numerical scale of the variables.

# ## 3.6 Data Integration
# 
# Data integration is not required for this project. The analysis is based on a single structured dataset in which all predictor variables and the diagnostic label are already contained within the same table.
# 
# Therefore, no merging or appending of additional data sources is necessary.

# ## 3.7 Data Formatting
# 
# The dataset is already available in a format compatible with classical machine learning algorithms. The predictor variables are organized in a numerical feature matrix (`X`) and the diagnosis labels in a binary target vector (`y`).
# 
# After applying the train–test split and feature scaling, the dataset structure is compatible with the machine learning workflow. [5]

# ## 3.8 Final Dataset for Modelling
# 
# After completing the data preparation steps, the dataset is ready for the modeling phase.
# 
# The preprocessing pipeline produced standardized training and test datasets that will serve as input for the machine learning models. The resulting data objects are:
# 
# - **X_train_scaled**: standardized training feature matrix  
# - **X_test_scaled**: standardized test feature matrix  
# - **y_train**: training target labels  
# - **y_test**: test target labels

# ## 3.9 Export of Processed Data
# 
# To ensure reproducibility and allow consistent data access across notebooks, the processed datasets are exported as CSV files.
# 
# The standardized training and test feature matrices as well as the corresponding target vectors are stored in the `data/processed` directory. This separation between raw and processed data supports a transparent and reproducible workflow.

output_dir = Path("../data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

X_train_scaled_df.to_csv(output_dir / "X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv(output_dir / "X_test_scaled.csv", index=False)

y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)


# ## References
# 
# [1] IBM Corporation (2011): *IBM SPSS Modeler CRISP-DM Guide*
# 
# [2] Street, W. N.; Wolberg, W. H.; Mangasarian, O. L. (1993): *Nuclear feature extraction for breast tumor diagnosis*
# 
# [3] James, G.; Witten, D.; Hastie, T.; Tibshirani, R. (2023): *An Introduction to Statistical Learning with Applications in Python*
# 
# [4] Sidey-Gibbons, J. A. M.; Sidey-Gibbons, C. J. (2019): *Machine learning in medicine: a practical introduction*
# 
# [5] Müller, A. C.; Guido, S. (2016): *Introduction to Machine Learning with Python*
