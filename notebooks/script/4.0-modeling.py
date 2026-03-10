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

# # 4. Modelling
# 
# This section focuses on selecting, training, and evaluating machine learning models
# to classify breast cancer cases as malignant or benign.
# 
# The modelling phase follows the CRISP-DM methodology and includes model selection,
# training, hyperparameter tuning, and performance evaluation.

# Data handling
import pandas as pd
from pathlib import Path

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)


data_dir = Path("../data/processed")

X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
X_test = pd.read_csv(data_dir / "X_test_scaled.csv")

y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")


print("Training features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Test labels:", y_test.shape)


# ## 4.1 Model Selection
# 
# Several machine learning models are evaluated to determine which approach
# performs best for the breast cancer classification task.
# 
# The selected models represent different machine learning paradigms:
# 
# - **Logistic Regression** – A linear probabilistic model commonly used for binary classification.
# - **Random Forest** – An ensemble method based on multiple decision trees.
# - **Support Vector Machine (SVM)** – A margin-based classifier that finds an optimal decision boundary.
# - **K-Nearest Neighbours (KNN)** – A distance-based model that classifies samples based on the nearest training instances.
# 
# Using different types of models allows us to compare their performance
# and better understand which approach is most suitable for this dataset.

# ## 4.2 Baseline Model
# 
# Before training machine learning models, a baseline classifier is created.
# The baseline model predicts the most frequent class in the training data.
# 
# This provides a simple reference point for later model comparison.
# A useful machine learning model should perform better than this naive approach.

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline)

print("Baseline Model Performance")
print("Accuracy:", round(baseline_accuracy, 4))
print("Precision:", round(baseline_precision, 4))
print("Recall:", round(baseline_recall, 4))
print("F1-score:", round(baseline_f1, 4))


# The baseline model predicts only the majority class and therefore
# does not provide meaningful diagnostic capability.
# 
# While the accuracy may appear moderate due to class imbalance,
# the model fails to capture the underlying patterns in the data.
# 
# More advanced machine learning models should significantly
# outperform this baseline.

# ## 4.3 Logistic Regression
# 
# Logistic Regression is used as a baseline machine learning model for
# binary classification. The model estimates the probability that a sample
# belongs to a particular class using a logistic function.

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)

print("Logistic Regression Performance")
print("Accuracy:", round(log_accuracy, 4))
print("Precision:", round(log_precision, 4))
print("Recall:", round(log_recall, 4))
print("F1-score:", round(log_f1, 4))

y_prob_log = log_model.predict_proba(X_test)


# The logistic regression model achieves very high classification performance
# with an accuracy of 0.9825 on the test set.
# 
# This result indicates that the morphological features extracted from
# the cell nuclei provide strong predictive information for distinguishing
# between malignant and benign tumours.
# 
# The strong performance of a linear classifier suggests that the classes
# are relatively well separable in the feature space.

# ## 4.4 Random Forest
# 
# Random Forest is an ensemble learning method that combines multiple
# decision trees to improve predictive performance and reduce overfitting.

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("Random Forest Performance")
print("Accuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1-score:", round(rf_f1, 4))

feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top_features = feature_importances.head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# The Random Forest model achieves strong predictive performance,
# confirming that ensemble methods are highly effective for tabular
# classification tasks.
# 
# The feature importance analysis indicates that measurements related
# to tumour size, such as radius, perimeter, and area, are among the
# most influential predictors for distinguishing malignant and benign
# cases.
# 
# This observation is consistent with the exploratory data analysis,
# which showed strong correlations between these features and the
# diagnostic outcome.

# ## 4.5 Support Vector Machine (SVM)
# 
# Support Vector Machines aim to find the optimal hyperplane that
# maximally separates the classes in feature space.

svm_model = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print("SVM Performance")
print("Accuracy:", round(svm_accuracy, 4))
print("Precision:", round(svm_precision, 4))
print("Recall:", round(svm_recall, 4))
print("F1-score:", round(svm_f1, 4))


# The Support Vector Machine model achieves classification performance
# very similar to logistic regression.
# 
# This suggests that the dataset is largely linearly separable, meaning
# that a simple linear decision boundary is already sufficient to
# distinguish between malignant and benign cases.
# 
# In such situations, more complex models do not necessarily lead to
# substantial improvements in predictive performance.

# ## 4.6 K-Nearest Neighbours (KNN)
# 
# KNN is a distance-based algorithm that classifies samples based on
# their proximity to neighbouring training samples in feature space.

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)

print("KNN Performance")
print("Accuracy:", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall:", round(knn_recall, 4))
print("F1-score:", round(knn_f1, 4))


# The KNN model achieves strong classification performance, indicating
# that similar tumour cases are located close to each other in the feature space.
# 
# This result confirms that the scaled diagnostic features provide a meaningful
# representation for distance-based classification.
# 
# However, KNN may be more sensitive to the choice of hyperparameters and
# the local structure of the data than some of the other evaluated models.

# ## 4.7 Hyperparameter Tuning

# ## 4.8 Model Comparison
# 
# The results show the performance differences between the evaluated models.
# More advanced models such as Random Forest and SVM typically achieve
# higher predictive performance compared to simpler baseline models.
# 
# The comparison allows us to identify the most suitable model for
# the breast cancer classification task.

results = pd.DataFrame({

    "Model": [
        "Baseline",
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN"
    ],

    "Accuracy": [
        baseline_accuracy,
        log_accuracy,
        rf_accuracy,
        svm_accuracy,
        knn_accuracy
    ],

    "Precision": [
        baseline_precision,
        log_precision,
        rf_precision,
        svm_precision,
        knn_precision
    ],

    "Recall": [
        baseline_recall,
        log_recall,
        rf_recall,
        svm_recall,
        knn_recall
    ],

    "F1 Score": [
        baseline_f1,
        log_f1,
        rf_f1,
        svm_f1,
        knn_f1
    ]
})

results = results.sort_values("Accuracy", ascending=False)
results.reset_index(drop=True, inplace=True)
results

