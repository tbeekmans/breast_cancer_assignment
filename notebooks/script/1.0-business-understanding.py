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
# Breast cancer is one of the most common types of cancer worldwide, and early diagnosis plays an important role in improving treatment outcomes. In clinical practice, diagnostic procedures often involve analysing tissue samples in order to determine whether a tumor is malignant or benign.
# 
# Machine learning methods have increasingly been applied in medical diagnostics to support clinical decision-making and improve the consistency of diagnostic processes [2]. In such scenarios, models are typically used as decision-support tools rather than as standalone diagnostic systems.
# 
# The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. It contains numerical features extracted from digitized images of fine needle aspirate (FNA) samples of breast masses. These features describe morphological characteristics of cell nuclei, such as size, shape, and texture.
# 
# According to Street et al. [1], these measurements capture structural differences between benign and malignant tumors and can therefore support diagnostic classification.
# 
# This project investigates whether classical machine learning models can accurately classify breast cancer cases based on these extracted features.

# ## 1.2 Problem Statement
# 
# The objective of this project is to determine whether the available diagnostic features can be used to accurately classify breast tumors as malignant or benign using machine learning methods.
# 
# This task represents a supervised binary classification problem in which each observation contains numerical features describing characteristics of cell nuclei, and the target variable indicates whether the tumor is malignant or benign.
# 
# Reliable classification is particularly important in medical contexts because incorrect predictions may have serious consequences. In particular, predicting a malignant tumor as benign (a false negative) may delay necessary medical treatment. For this reason, model evaluation should consider not only overall accuracy but also metrics such as recall and precision.
# 
# The project focuses on applying classical machine learning techniques to investigate how well the dataset supports this classification task. The analysis follows the CRISP-DM methodology, which structures data science projects into iterative phases including business understanding, data understanding, data preparation, modelling, and evaluation [3].
# 
# The resulting workflow should therefore provide a reproducible and transparent classification process based on the available diagnostic features.

# ## 1.3 Stakeholders
# 
# Academic stakeholders:
# 
# - The student conducting the analysis as part of the course assignment
# - Teaching staff responsible for evaluating the project
# 
# Domain stakeholders:
# 
# - Healthcare professionals who may benefit from decision-support tools in medical diagnostics
# - Patients who are affected by diagnostic decisions and treatment planning

# ## 1.4 Business Objectives and Success Criteria
# 
# | ID  | Business Objective | Success Criteria |
# |-----|-------------------|------------------|
# | BO1 | Investigate whether diagnostic features allow reliable classification of breast tumors | A machine learning approach can distinguish malignant and benign tumors with consistently strong performance on unseen data |
# | BO2 | Develop a transparent and reproducible data science workflow | The full analysis workflow can be executed end-to-end using the provided notebooks |
# | BO3 | Provide interpretable insights into relevant diagnostic features | The analysis identifies features that contribute strongly to tumor classification |

# ## 1.5 Data Mining Goals and Success Criteria
# 
# | ID   | Data Mining Goal | Success Criteria |
# |------|------------------|------------------|
# | DMG1 | Explore and understand the dataset | Data distributions, correlations, and data quality issues are analysed and documented |
# | DMG2 | Prepare the dataset for machine learning | Data is cleaned, structured, and suitable for model training |
# | DMG3 | Train and compare several classification models | Multiple classical machine learning models are trained and evaluated |
# | DMG4 | Select and justify a final model | The final model is selected based on accuracy, precision, recall, F1-score, confusion matrix results, and overall suitability for the problem |

# ## 1.6 Inventory of Resources
# 
# This project is conducted as an individual assignment within the DataOps specialisation.
# 
# **Personnel:**
# - One student responsible for the complete workflow
# 
# **Data Resources:**
# - Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository [1]
# 
# **Hardware and Infrastructure:**
# - Personal workstation for development and model training
# 
# **Software and Tools:**
# - Python 3.11
# - Jupyter Notebooks
# - Pandas, NumPy, Scikit-learn
# - Matplotlib and Seaborn for visualisation

# ## 1.7 Constraints and Assumptions
# 
# **Constraints:**
# 
# - The project must be completed individually within a limited timeframe
# - Only classical machine learning techniques may be used
# - Deep learning methods are excluded according to the assignment requirements
# - The project relies solely on the provided dataset
# 
# **Assumptions:**
# 
# - The dataset contains sufficient information to distinguish malignant and benign tumors
# - Classical machine learning models are suitable for this classification problem
# - The resulting model is intended as a decision-support tool and not as a standalone medical diagnosis system

# ## 1.8 Risks
# 
# | Risk | Impact | Contingency |
# |-----|------|-------------|
# | Overfitting due to small dataset size | Model may perform well on training data but poorly on unseen data | Use cross-validation and evaluate models on a hold-out test set |
# | Limited medical domain expertise | Misinterpretation of feature meaning | Focus on statistical interpretation rather than clinical conclusions |
# | Strong feature correlations | Some models may become unstable or biased | Apply feature selection or regularization techniques |

# ## 1.9 Project Plan
# 
# The timeline for this assignment follows the schedule described in the course manual [4].
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
# RQ1: Which features are most predictive for distinguishing between malignant and benign tumors?
# 
# RQ2: Which classical machine learning model achieves the best classification performance on this dataset?
# 
# RQ3: How do different evaluation metrics such as accuracy, precision, recall, and F1-score compare across the tested models?

# ## References
# 
# [1] Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE International Symposium on Electronic Imaging: Science and Technology. Retrieved from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# [2] J. A. M. Sidey-Gibbons and C. J. Sidey-Gibbons, “Machine learning in medicine: a practical introduction,” BMC Med Res Methodol, vol. 19, no. 1, p. 64, Dec. 2019, doi: 10.1186/s12874-019-0681-4
# 
# [3] “IBM SPSS Modeler CRISP-DM Guide”
# 
# [4] Saxion Brightspace: DataOps Specializaton - Module Website - Indiviual Assignment - Information & Rubric. Retrieved from https://data-ops-project-module-ed0669.gitlab.io/manual.html
