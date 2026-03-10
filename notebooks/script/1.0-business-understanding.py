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

# # 1. Business Understanding

# ## 1.1 Business Background
# 
# Breast cancer diagnosis commonly involves analysing tissue samples to determine whether a tumour is malignant or benign. One frequently used method is fine-needle aspiration (FNA), where cells from a breast mass are collected and examined microscopically. Quantitative features describing the morphology of the cell nuclei can be extracted from these samples and used for computational analysis of tumour characteristics. [2]
# 
# Machine learning methods are increasingly applied in medical research and clinical decision support. These algorithms can analyse complex datasets and identify patterns that may assist clinicians in diagnostic and predictive tasks. In medical contexts, such models are typically used as decision-support systems that complement clinical expertise rather than replacing it. [3]
# 
# Several studies have explored the use of machine learning techniques for cancer detection. Classification algorithms such as Support Vector Machines (SVM) and Artificial Neural Networks (ANN) have demonstrated strong predictive performance when applied to biomedical datasets, helping researchers identify patterns related to cancer diagnosis. [4]
# 
# Comparative studies of machine learning approaches applied to breast cancer datasets have reported promising results for multiple classification algorithms. Models such as logistic regression, support vector machines, and ensemble methods have achieved high accuracy when distinguishing between benign and malignant tumours based on diagnostic features. [5]
# 
# In this project, classical machine learning models are therefore applied to the Breast Cancer Wisconsin dataset in order to evaluate their ability to classify tumours based on these extracted diagnostic features.

# ## 1.2 Problem Statement
# 
# The objective of this project is to investigate whether diagnostic features extracted from breast tumour samples can be used to accurately classify tumours as malignant or benign using machine learning methods. This task represents a supervised binary classification problem in which each observation contains numerical features describing morphological characteristics of cell nuclei derived from fine needle aspiration (FNA) images, while the target variable indicates whether the tumour is malignant or benign. [2]
# 
# Reliable classification is particularly important in medical contexts because incorrect predictions may have serious consequences. In particular, predicting a malignant tumour as benign (a false negative) may delay necessary treatment and negatively affect patient outcomes. For this reason, model evaluation should consider not only overall accuracy but also metrics such as recall and precision when assessing diagnostic models. [3] To investigate this classification task, several classical machine learning algorithms are applied to the "Breast Cancer Wisconsin (Diagnostic)" dataset in order to evaluate how effectively these diagnostic features support tumour classification. [2]
# 
# The analysis follows the CRISP-DM methodology, which structures data science projects into iterative phases including business understanding, data understanding, data preparation, modelling, and evaluation. [6]
# 
# The resulting workflow aims to provide a reproducible and transparent pipeline for training and evaluating machine learning models for breast cancer classification.

# ## 1.3 Stakeholders
# 
# Domain Stakeholders:
# 
# - Healthcare professionals who analyze tumour samples and may benefit from decision-support tools in medical diagnostics
# - Patients who are affected by diagnostic decisions and treatment planning
# 
# Research Stakeholders:
# - Data scientists and researchers interested in evaluating machine learning approaches for medical datasets
# 
# Academic Stakeholders:
# 
# - Lecturer at Saxion University responsible for defining the assignment context and evaluating the project

# ## 1.4 Business Objectives and Success Criteria
# 
# | ID  | Business Objective | Success Criteria |
# |-----|-------------------|------------------|
# | BO1 | Support the analysis of breast tumour samples by exploring whether machine learning methods can assist in distinguishing malignant and benign cases | Machine learning models demonstrate reliable classification performance on unseen data |
# | BO2 | Improve understanding of which diagnostic features are most relevant for tumour classification | The analysis identifies features that show strong relationships with the diagnostic outcome |
# | BO3 | Develop a transparent and reproducible machine learning workflow | The full analysis workflow can be executed end-to-end using the provided notebooks |

# ## 1.5 Data Mining Goals and Success Criteria
# 
# | ID   | Data Mining Goal | Success Criteria |
# |------|------------------|------------------|
# | DMG1 | Analyse the dataset to identify relevant patterns, feature relationships, and predictive signals | Data distributions and correlations between features and the target variable are analysed and documented |
# | DMG2 | Prepare the dataset for machine learning modelling | Data is cleaned, structured, and preprocessed for model training |
# | DMG3 | Train and compare several classical machine learning models for tumour classification | Multiple models are trained and evaluated on the dataset |
# | DMG4 | Identify the most suitable model for the classification task | The selected model demonstrates strong performance across the chosen evaluation metrics |

# ## 1.6 Inventory of Resources
# 
# This project is conducted as an individual assignment within the DataOps specialisation.
# 
# **Personnel:**
# - Analyst: One student responsible for implementing the complete data science workflow
# - Academic Stakeholders: Course lecturers can be asked for advice during the project
# - Domain Stakeholders: Domain experts/practitioners who may be contacted to provide insights related to medical domain
# - Research Stakeholders: Data scientists who may be contacted to share knowledge related to medical dataset analysis
# 
# **Data Resources:**
# - Dataset: Breast Cancer Wisconsin (Diagnostic) dataset retrieved from the UCI Machine Learning Repository [1]
# 
# **Hardware and Infrastructure:**
# - Compute Environment: Personal workstation for data analysis and model training
# 
# **Software and Tools:**
# - Environment Management: Conda environment for dependency management and reproducibility
# - Programming Language: Python 3.11
# - Development Environment: Jupyter Notebooks for EDA
# - Data Analysis Libraries: Python libraries for data processing (e.g. pandas, numpy)
# - Machine Learning Frameworks: Libraries for implementing machine learning models (e.g. scikit-learn)
# - Visualization Tools: Libraries for data visualisation (e.g. matplotlib, seaborn)
# - Reference Management: Zotero to manage literature and citations

# ## 1.7 Constraints and Assumptions
# 
# **Constraints:**
# 
# - The project must be completed individually within a limited timeframe
# - Only classical machine learning techniques may be used
# - Deep learning methods are excluded according to the assignment requirements
# - The project relies solely on the provided dataset
# - The dataset size is relatively small (569 instances according to UCI Machine Learning Repository dataset description [1])
# 
# **Assumptions:**
# 
# - The dataset contains sufficient information to distinguish malignant and benign tumors
# - The dataset labels are correct and represent reliable diagnostic outcomes
# - Classical machine learning models are suitable for this classification problem

# ## 1.8 Risks
# 
# | Risk | Impact | Contingency |
# |-----|------|-------------|
# | Overfitting due to small dataset size | Model may perform well on training data but poorly on new samples | Use cross-validation and evaluate models on a separate test set |
# | Limited medical domain expertise | Incorrect interpretation of feature relevance or model results | Conduct literature research and focus on statistical interpretation rather than clinical conclusions |
# | Strong feature correlations | Some models may become unstable or biased | Apply feature selection or regularization techniques |

# ## 1.9 Project Plan
# 
# The timeline for this assignment follows the schedule described in the course manual and is aligned with the CRISP-DM framework [7]:
# 
# | Week | CRISP-DM Phase | Tasks |
# |-----|---------------|------|
# | 3.1–3.3 | Business Understanding | Define project context, review related literature, define objectives |
# | 3.1–3.3 | Data Understanding | Explore dataset structure, analyse distributions and correlations |
# | 3.1–3.3 | Data Preparation | Clean and preprocess the dataset |
# | 3.3–3.5 | Modelling | Train and tune several machine learning models |
# | 3.3–3.5 | Evaluation | Compare models using appropriate evaluation metrics |

# ## 1.10 Research Questions
# 
# The following research questions guide the analysis in this project:
# 
# RQ1: Which diagnostic features show the strongest relationship with the tumour diagnosis (malignant or benign)?
# 
# RQ2: How accurately can classical machine learning models classify breast tumours based on the available diagnostic features?
# 
# RQ3: Which machine learning model provides the most reliable classification performance for this dataset?

# ## References
# 
# [1] UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Dataset. Retrieved from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# [2] Street, W. N.; Wolberg, W. H.; Mangasarian, O. L. (1993): Nuclear feature extraction for breast tumor diagnosis
# 
# [3] Sidey-Gibbons J. A. M.; Sidey-Gibbons, C. J. (2019): Machine learning in medicine: a practical introduction
# 
# [4] Sharma, A.; Kulshrestha S.; Daniel, S. (2018): Machine Learning Approaches for Cancer Detection
# 
# [5] Rovshenov, A; Peker, S. (2022): Performance Comparison of Different Machine Learning Techniques for Early Prediction of Breast Cancer using Wisconsin Breast Cancer Dataset
# 
# [6] IBM Corporation (2011): IBM SPSS Modeler CRISP-DM Guide
# 
# [7] Saxion Brightspace: DataOps Specializaton - Module Website - Indiviual Assignment - Information & Rubric. Retrieved from https://data-ops-project-module-ed0669.gitlab.io/manual.html
