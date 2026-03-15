# DataOps Specialization – Individual Assignment - Breast Cancer Classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
<img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Goal:** Develop a machine learning model to accurately classify breast cancer cases as malignant or benign.

---

# Environment Setup

Create the Conda environment:

conda env create -f environment.yml  
conda activate breast_cancer_assignment  

Install the local package:

pip install -e .

Start Jupyter Lab:

jupyter lab

---

# Notebook Structure

The CRISP-DM workflow is implemented through the following notebooks:

1. 1_business_understanding.ipynb  
2. 2_data_understanding.ipynb  
3. 3_data_preparation.ipynb  
4. 4_modeling.ipynb  
5. 5_evaluation.ipynb  

These notebooks should be executed in this order.

Each notebook contains both:

- Markdown explanations and interpretations  
- executable Python code  

to document the complete analysis workflow.

---

# Data

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset from UCI Machine Learning Repository.

Dataset structure within the project:

```
data/
├── raw
│   └── dataset.csv
└── processed
    ├── X_train_scaled.csv
    ├── X_test_scaled.csv
    ├── y_train.csv
    └── y_test.csv
```

- raw contains the original dataset  
- processed contains the datasets generated during preprocessing and used for modeling  

---

# References

The `references/` directory contains academic papers and literature used during the project.  

---

# Project Organization

This project was created using the Cookiecutter Data Science template.  
Some directories exist as part of the template structure but are not actively used in the assignment.

```
breast_cancer_assignment/
├── Makefile
├── README.md
├── environment.yml
├── pyproject.toml
│
├── data
│   ├── external
│   ├── interim
│   ├── raw
│   │   └── dataset.csv
│   └── processed
│       ├── X_train_scaled.csv
│       ├── X_test_scaled.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── docs
│   ├── mkdocs.yml
│   ├── README.md
│   └── docs
│       ├── getting-started.md
│       └── index.md
│
├── models
│
├── notebooks  
│   ├── 1_business_understanding.ipynb  
│   ├── 2_data_understanding.ipynb  
│   ├── 3_data_preparation.ipynb  
│   ├── 4_modeling.ipynb  
│   ├── 5_evaluation.ipynb 
│   └── script
│       ├── 1_business_understanding.py 
│       ├── 2_data_understanding.py 
│       ├── 3_data_preparation.py 
│       ├── 4_modeling.py 
│       └── 5_evaluation.py
│
├── references
│
├── reports
│   └── figures
│
├── tests
│   └── test_data.py
│
└── breast_cancer_assignment
    │
    ├── __init__.py
    │
    ├── config.py
    │
    ├── dataset.py
    │
    ├── features.py
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py         
    │   └── train.py
    │
    └── plots.py
```

---

# Acknowledgements

Generative AI tools were used during the preparation of this assignment to support language refinement and occasionally for code clarification or debugging. All analytical decisions, implementations, and interpretations were developed by the author based on the knowledge gained during course workshops and literature research.
