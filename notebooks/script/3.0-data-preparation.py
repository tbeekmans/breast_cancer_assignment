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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from breast_cancer_assignment.dataset import load_data

# Notebook settings
pd.set_option("display.max_columns", None)

# Load dataset
df = load_data()


# # 3.0 Data Preparation
# 
# The data preparation phase transforms the dataset into a form suitable for machine learning modelling. According to the CRISP-DM methodology, this phase focuses on preparing the dataset so that it can be reliably used for training and evaluating predictive models [1].
# 
# Based on the findings from the data understanding phase, the dataset does not require corrective cleaning such as missing value imputation or duplicate removal. Instead, the preparation focuses on organising the predictor and target variables, splitting the dataset into training and test subsets, and applying preprocessing steps such as feature scaling where required for the selected machine learning algorithms [3].

# ## 3.1 Data Selection
# 
# All observations and predictor variables are retained for modelling. The previous data quality analysis did not identify missing values, duplicate observations, or invalid measurements that would justify excluding records from the dataset [1].
# 
# Although the data understanding phase revealed strong correlations between several variables, the full predictor set is preserved in order to retain the complete diagnostic information available in the dataset [3].
# 
# The variable `target` represents the tumour diagnosis and is used as the response variable for the supervised classification task.

# Define feature matrix and target variable
X = df.drop(columns="target")
y = df["target"]

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# The resulting feature matrix contains 569 observations and 30 predictor variables, while the target vector contains 569 binary diagnostic labels. This structure is suitable for supervised machine learning classification tasks [2].

print("Selected feature columns:")
print(X.columns.tolist())


# The selected predictors include all diagnostic measurements provided in the dataset. No attributes are excluded, ensuring that the complete set of morphological features is available for model training.

# ## 3.2 Data Cleaning
# 
# The data understanding phase included a comprehensive data quality assessment. This analysis confirmed that the dataset does not contain missing values, duplicate observations, or invalid numerical measurements. Therefore, no imputation or record removal is required before modelling [1].
# 
# Potential outliers were identified during exploratory analysis. However, these observations are retained because extreme values in biomedical datasets may represent genuine biological variation rather than measurement errors [4].
# 
# Consequently, no observations or attributes are removed during the data cleaning phase.

# ## 3.3 Data Construction
# 
# No additional attributes or records are constructed. The dataset already consists of domain-specific diagnostic measurements extracted from fine needle aspirate (FNA) images of breast masses, which were designed to support tumour classification tasks [2].
# 
# Because these variables already encode relevant geometric and structural characteristics of tumour cells, further feature construction is not required for the modelling workflow.

# ## 3.4 Train–Test Split
# 
# To evaluate machine learning models objectively, the dataset must be divided into separate training and test subsets.  
# The training data is used to fit the machine learning models, while the test data is reserved for evaluating final model performance on previously unseen observations. This separation helps prevent overly optimistic performance estimates and allows a more realistic assessment of how well a model generalizes to new data [3].
# 
# In classification problems, it is also important to preserve the original class distribution when splitting the dataset. If the proportion of malignant and benign tumours changes significantly between training and test sets, the evaluation results may become biased. For this reason, a **stratified sampling strategy** is used. Stratified splitting ensures that both subsets maintain approximately the same class proportions as the original dataset [5].
# 
# The dataset is divided using an **80/20 split**, where 80% of the data is used for model training and 20% is reserved for testing. This ratio is commonly used in machine learning projects because it provides sufficient training data while keeping an independent subset for evaluation [5].
# 
# A fixed random seed (`random_state = 42`) is used to ensure that the data split is reproducible. Reproducibility is an important aspect of scientific workflows and allows the results to be replicated consistently [1].

# Perform train-test split

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


# The dataset is divided into training and test subsets using an 80/20 split.
# 
# The output confirms that 455 observations are assigned to the training set and 114 observations to the test set. The training data is used to fit the machine learning models, while the test data is reserved for evaluating model performance on previously unseen observations [3].
# 
# Stratified sampling ensures that the class distribution remains consistent across both subsets. The proportions of benign and malignant tumours are therefore preserved in both the training and test sets, supporting reliable model evaluation [5].

print("Training class distribution:")
print(y_train.value_counts(normalize=True))

print("Test class distribution:")
print(y_test.value_counts(normalize=True))


# The class distribution in the training and test sets is nearly identical, confirming that stratified sampling preserved the original class balance of the dataset. Maintaining similar class proportions helps prevent biased evaluation results in classification tasks [3].

# ## 3.5 Feature Scaling
# 
# The predictor variables are standardized using `StandardScaler`. Standardization transforms each feature to have a mean of approximately 0 and a standard deviation of approximately 1, ensuring that all variables operate on a comparable numerical scale [3].
# 
# The scaler is fitted only on the training data and then applied to both the training and test sets. This approach prevents data leakage and ensures that information from the test data does not influence the training process [5].

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
# The transformation does not change the number of observations or features; it only adjusts the numerical scale of the variables.

# ## 3.6 Data Integration
# 
# Data integration is not required for this project. The analysis is based on a single structured dataset in which all predictor variables and the diagnostic label are already contained within the same table [1].
# 
# Therefore, no merging or appending of additional data sources is necessary.

# ## 3.7 Data Formatting
# 
# The dataset is already available in a format compatible with classical machine learning algorithms. The predictor variables are organised in a numerical feature matrix (`X`) and the diagnosis labels in a binary target vector (`y`).
# 
# After applying the train–test split and feature scaling, the dataset structure is fully compatible with the modelling algorithms used in this project [5].

# ## 3.8 Final Dataset for Modelling
# 
# After completing the data preparation steps, the dataset is ready for the modelling phase.
# 
# The preprocessing pipeline produced standardized training and test datasets that will serve as the input for the machine learning models. The resulting data objects are:
# 
# - **X_train_scaled** – standardized training feature matrix  
# - **X_test_scaled** – standardized test feature matrix  
# - **y_train** – training target labels  
# - **y_test** – test target labels  
# 
# The training data will be used to train and tune the machine learning models, while the test dataset will remain unseen during model development and will only be used for final performance evaluation.
# 
# This separation ensures that model performance is evaluated on previously unseen observations, providing a more reliable estimate of generalization performance.

print("Training features:", X_train_scaled.shape)
print("Test features:", X_test_scaled.shape)
print("Training labels:", y_train.shape)
print("Test labels:", y_test.shape)


# ## 3.9 Export of Processed Data
# 
# To ensure reproducibility and allow consistent data access across the modelling and evaluation notebooks, the processed datasets are exported as CSV files.
# 
# The standardized training and test feature matrices as well as the corresponding target vectors are stored in the `data/processed` directory. This separation between raw and processed data supports a transparent and reproducible workflow.

from pathlib import Path

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
# [1] IBM Corporation. *IBM SPSS Modeler CRISP-DM Guide*. 2011.
# 
# [2] W. N. Street, W. H. Wolberg, and O. L. Mangasarian.  
# "Nuclear feature extraction for breast tumor diagnosis." 1993.
# 
# [3] G. James, D. Witten, T. Hastie, and R. Tibshirani.  
# *An Introduction to Statistical Learning*. Springer, 2013.
# 
# [4] J. A. M. Sidey-Gibbons and C. J. Sidey-Gibbons.  
# "Machine learning in medicine: a practical introduction." *BMC Medical Research Methodology*, 2019.
# 
# [5] A. C. Müller and S. Guido.  
# *Introduction to Machine Learning with Python*. O’Reilly Media, 2016.
