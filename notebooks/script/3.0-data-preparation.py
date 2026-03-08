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

# ## 3.1 Feature Selection
# 
# Before training machine learning models, the dataset must be prepared by defining the predictor variables and the target variable.
# 
# The dataset contains numerical measurements describing morphological characteristics of cell nuclei. These measurements serve as the input features for the classification task.
# 
# The diagnostic label indicates whether a tumour is malignant or benign and is therefore used as the target variable.

# Define feature matrix and target variable

X = df.drop(columns="target")
y = df["target"]

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# The resulting feature matrix contains 30 numerical predictor variables, while the target vector contains the binary diagnosis label.
# 
# At this stage, all original predictor variables are retained. Feature selection in the sense of removing redundant or less informative variables is not yet performed, since the initial goal is to preserve the full dataset structure for model comparison.
# 
# Potential feature redundancy identified in the data understanding phase will be taken into account later during modelling and evaluation.

# ## 3.2 Train–Test Split
# 
# To evaluate the performance of machine learning models objectively, the dataset is divided into separate training and test sets.
# 
# The training set is used to train the models, while the test set is reserved for evaluating model performance on previously unseen data. This separation helps prevent overly optimistic performance estimates.
# 
# A stratified split is applied to ensure that the class distribution of malignant and benign tumours remains consistent in both subsets.

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# The dataset is split into training and test sets using an 80/20 ratio.
# 
# Stratified sampling ensures that the proportion of malignant and benign cases remains consistent across both subsets. This is particularly important in classification problems where class imbalance may affect model performance.
# 
# The resulting training set will be used for model training and hyperparameter tuning, while the test set will serve as an independent dataset for final model evaluation.

# ## 3.3 Feature Scaling
# 
# The dataset contains numerical features with substantially different scales. For example, variables describing tumour size such as `area` and `perimeter` have much larger numerical values than variables such as `smoothness` or `fractal_dimension`.
# 
# Machine learning algorithms that rely on distance calculations or gradient optimisation can be sensitive to such differences in feature scale. Without scaling, variables with larger magnitudes may dominate the learning process.
# 
# To address this issue, feature scaling is applied using the **StandardScaler**, which standardises each feature to have a mean of 0 and a standard deviation of 1.
# 
# The scaler is fitted only on the training data to avoid data leakage. The same transformation is then applied to both the training and test sets.

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply same transformation to test data
X_test_scaled = scaler.transform(X_test)

print("Scaled training set shape:", X_train_scaled.shape)
print("Scaled test set shape:", X_test_scaled.shape)

# Convert scaled training data to DataFrame for inspection
scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
print(scaled_df.describe().round(2))


# The scaling process standardises the feature distributions by transforming them to a mean of approximately 0 and a standard deviation of approximately 1.
# 
# The inspection confirms that the scaling procedure was applied correctly. Performing the fit operation exclusively on the training data prevents information from the test set from influencing the transformation, thereby avoiding data leakage.
# 
# The scaled datasets will be used for training machine learning models in the subsequent modelling phase.

# ## 3.4 Final Dataset for Modelling
# 
# After completing the preparation steps, the dataset is ready for the modelling phase.
# 
# The original dataset has been divided into training and test sets to enable an unbiased evaluation of model performance. Feature scaling has been applied to ensure that all numerical variables operate on a comparable scale.
# 
# The resulting datasets are:
# 
# - **X_train_scaled**: scaled training feature matrix  
# - **X_test_scaled**: scaled test feature matrix  
# - **y_train**: training target labels  
# - **y_test**: test target labels  
# 
# The training data will be used to train and tune machine learning models, while the test set will remain untouched during model development and will only be used for final performance evaluation.
# 
# These prepared datasets form the input for the modelling phase of the CRISP-DM process.

print("Training features:", X_train_scaled.shape)
print("Test features:", X_test_scaled.shape)
print("Training labels:", y_train.shape)
print("Test labels:", y_test.shape)

