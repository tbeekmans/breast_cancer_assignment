#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Breast Cancer Classification
# 
# Author: Tobias Beekmans  
# Master ICT – Software Engineering  
# DataOps Specialisation Project – Individual Assignment  
# Submission Date: 15.03.2026

import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score)


data_dir = Path("../data/processed")

X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
X_test = pd.read_csv(data_dir / "X_test_scaled.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")


# # 5. Evaluation
# 
# Following the CRISP-DM methodology, the evaluation phase assesses whether the trained models meet the objectives defined in the earlier phases and whether the obtained results are suitable for the classification task. [1]
# 
# In this project, the evaluation focuses on analysing classification errors, comparing model discrimination performance, and selecting the most suitable final model for breast cancer diagnosis.

log_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)


# ## 5.1 Evaluation Objectives
# 
# The purpose of the evaluation phase is to assess whether the trained models meet the objectives defined in the earlier phases and to determine which model is most suitable for the breast cancer classification task.
# 
# In a medical diagnostic context, evaluation must consider not only overall accuracy but also the ability to correctly identify malignant tumours. False negatives are particularly critical because incorrectly classifying a malignant tumour as benign may delay necessary treatment.
# 
# The evaluation therefore focuses on confusion matrices, ROC curves, AUC values, and the overall balance between predictive performance and interpretability.

# ## 5.2 Confusion Matrix
# 
# The confusion matrices provide a detailed view of the classification errors made by each model.

def plot_confusion_matrix(y_true, y_pred, title):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    plt.show()

plot_confusion_matrix(y_test, y_pred_log, "Confusion Matrix – Logistic Regression")
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix – Random Forest")
plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix – SVM")
plot_confusion_matrix(y_test, y_pred_knn, "Confusion Matrix – KNN")


# Logistic Regression and Support Vector Machine show very similar error patterns and produce only a small number of misclassifications. In particular, both models correctly identify most malignant and benign cases, which is important in a medical diagnosis setting.
# 
# Random Forest and K-Nearest Neighbours also perform well, but they show slightly more classification errors than the two best-performing models. From a clinical perspective, false negatives are especially important because malignant tumours should not be missed. The confusion matrices indicate that Logistic Regression and SVM handle this requirement particularly well.

# ## 5.3 ROC Curve and AUC
# 
# The ROC curves compare the discrimination ability of the evaluated models across different classification thresholds. A model with a curve closer to the top-left corner indicates stronger class separation, while the AUC summarizes this performance in a single value.

y_prob_log = log_model.predict_proba(X_test)[:,1]
y_prob_rf = rf_model.predict_proba(X_test)[:,1]
y_prob_svm = svm_model.predict_proba(X_test)[:,1]
y_prob_knn = knn_model.predict_proba(X_test)[:,1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)

auc_log = roc_auc_score(y_test, y_prob_log)
auc_rf = roc_auc_score(y_test, y_prob_rf)
auc_svm = roc_auc_score(y_test, y_prob_svm)
auc_knn = roc_auc_score(y_test, y_prob_knn)

plt.figure(figsize=(8,6))

plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={auc_log:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f})")
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc_knn:.3f})")

plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.show()


# Logistic Regression and Support Vector Machine achieve the strongest ROC performance, with curves that remain closest to the upper-left corner and the highest AUC values. Random Forest and K-Nearest Neighbours also show strong discrimination ability, but their ROC curves lie slightly below those of the best-performing models.
# 
# These results confirm that the diagnostic features provide strong predictive information for distinguishing benign and malignant tumours and that Logistic Regression and SVM are the most effective classifiers in this experiment.

# ## 5.4 Assessment Against Objectives
# 
# The evaluation results indicate that the modelling phase successfully addressed the main objective of the project: classifying breast tumours as malignant or benign using diagnostic features extracted from fine needle aspiration images.
# 
# All evaluated machine learning models clearly outperform the baseline classifier, showing that the dataset contains strong predictive signals. Logistic Regression and Support Vector Machine achieve the strongest overall performance, while Random Forest and K-Nearest Neighbours also provide reliable classification results.
# 
# These findings support the earlier data understanding results, which suggested that tumour size and boundary irregularity contain strong diagnostic information. The evaluation therefore confirms that classical machine learning methods are suitable for this classification task and that the prepared dataset supports accurate tumour classification.

# ## 5.5 Final Model Selection
# 
# Considering the evaluation results, Logistic Regression is selected as the final model for this project.
# 
# Logistic Regression achieves the highest classification performance, matching the Support Vector Machine in accuracy while also showing very strong precision, recall, and F1-score. At the same time, Logistic Regression offers greater interpretability than SVM, because the relationship between the predictors and the classification outcome can be more directly examined through the model coefficients.
# 
# This creates an important trade-off: while both models perform equally well in predictive terms, Logistic Regression provides a simpler and more transparent solution. In a medical decision support context, such transparency can be advantageous because model behaviour is easier to explain and justify.
# 
# The final evaluation therefore suggests that Logistic Regression provides the most suitable balance between predictive performance and interpretability for the breast cancer classification task.

# ## References
# 
# [1] IBM Corporation (2011): *IBM SPSS Modeler CRISP-DM Guide*
# 
# [2] Müller, A. C.; Guido, S. (2016): *Introduction to Machine Learning with Python*
# 
# [3] Rovshenov, A.; Peker, S. (2022): *Performance Comparison of Different Machine Learning Techniques for Early Prediction of Breast Cancer using Wisconsin Breast Cancer Dataset*
