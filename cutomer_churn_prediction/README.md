# Customer Churn Prediction

A machine learning project focused on understanding and predicting customer churn using customer demographic and subscription-related information.

The project is developed as an end-to-end machine learning portfolio project, featuring a reproducible engineering preprocessing pipeline, decoupled feature engineering modules, and an interactive Streamlit application.

> **Project Status:** Production-Ready — Data processing, multi-model benchmarking, inference pipelines, and Streamlit app code have been completely finalized.

---

## 1. Problem Statement

Customer churn occurs when customers stop using a company's products or services.

Being able to identify customers who are more likely to churn can help businesses understand customer behavior and take preventive actions.

This project aims to explore customer data, identify patterns associated with churn, and develop a machine learning model capable of predicting whether a customer is likely to churn.

---

## 2. Objectives

The main objectives of this project are to:

* Understand the structure and characteristics of the customer dataset.
* Perform exploratory data analysis (EDA).
* Identify patterns and relationships associated with customer churn.
* Handle data anomalies (such as dropping 11 null values) and drop redundant features (TotalCharges) to prevent multicollinearity.
* Select relevant features for machine learning.
* Perform nominal (One-Hot) and ordinal feature engineering without introducing data leakage.
* Train and compare multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost).
* Evaluate models using appropriate classification metrics (Accuracy, Precision, Recall, and F1-Score).
* Select a suitable final model based on balanced metrics.
* Build a reproducible machine learning pipeline artifact.
* Develop a standalone Streamlit web application for customer churn prediction.

---

## 3. Project Workflow

The project follows the following workflow:

```text
Problem Definition
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering (.ravel() 1D tracking)
        ↓
Train/Test Split
        ↓
Preprocessing (Null handling)
        ↓
Model Training (5 Algorithms compared)
        ↓
Model Evaluation & Selection
        ↓
Production Pipeline
        ↓
Streamlit Application
```

The project is currently completed through all development stages.

---

## 4. Dataset

The dataset contains customer-level information used to analyze factors associated with customer churn.

### Feature Attributes
* gender: Customer gender (Male / Female)
* Dependents: Whether the customer has dependents (Yes / No)
* Contract: Active subscription length (month-to-month / One year / Two year)
* tenure: Number of months the customer has stayed with the company
* MonthlyCharges: The amount charged to the customer monthly
* Churn: Target variable representing whether a customer churned

### Target Variable
**Churn**
* 1 (Yes) — Customer churned
* 0 (No) — Customer did not churn

---

## 5. Data Understanding & Key Insights

The initial phase focused on auditing data quality and uncovering distribution patterns before applying machine learning techniques. 

### Core Architectural Decisions & Insights:
* Null Value Handling: 11 missing values were isolated and dropped cleanly during training to protect dataset consistency.
* Multicollinearity Removal: EDA revealed a strong positive correlation of 0.82 between tenure and TotalCharges. Since TotalCharges is highly redundant (essentially a direct calculation of tenure × MonthlyCharges), it was permanently dropped. This eliminates multicollinearity and keeps our linear model boundaries stable.
* Dimensional Vector Adjustments: During notebook testing, OneHotEncoder shape formatting threw a ValueError by outputting a 2D matrix back into a 1D column. This was resolved using `.ravel()` to flatten transformed training arrays into clean sequences.

---

## 6. Model Performance & Evaluation

Five machine learning models were developed cell-by-cell in our experimental notebook and evaluated side-by-side using the validation matrix splits. 

### Final Benchmark Results Table:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **79.67%** | **65.60%** | **48.97%** | **56.08%** |
| XGBoost | 78.04% | 60.00% | 51.37% | 55.35% |
| Support Vector Machine | 80.04% | 71.43% | 41.10% | 52.17% |
| Random Forest | 76.13% | 56.12% | 45.55% | 50.28% |
| Decision Tree | 72.50% | 48.24% | 51.71% | 49.92% |

### Selection Strategy:
Logistic Regression was chosen as our final deployment model. While Decision Tree achieved a slightly higher Recall (51.71%), its Precision was poor (48.24%). Logistic Regression achieved the highest overall F1-Score (56.08%), combined with high Precision (65.60%). This ensures fewer false-positive errors, saving resources during proactive customer retention campaigns.

---

## 7. Project Structure

The project directory structure is organized as follows:

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── prediction.py
│
├── models/
│   └── churn_pipeline.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 8. Development Progress Tracking

| Stage | Status |
| --- | --- |
| Problem Definition | Completed |
| Data Understanding | Completed |
| Exploratory Data Analysis | Completed |
| Feature Engineering | Completed |
| Train/Test Split | Completed |
| Preprocessing | Completed |
| Model Training | Completed |
| Model Evaluation | Completed |
| Model Selection | Completed |
| Production Pipeline | Completed |
| Streamlit Application | Completed |

---

## 9. Modular File Responsibilities

1. **src/data_preprocessing.py**: Formats string text anomalies into clean lowercases and handles column structural dropping rules.
2. **src/feature_engineering.py**: Builds a scikit-learn ColumnTransformer to route nominal categories (gender, Dependents) through a OneHotEncoder(drop='first'), maps Contract values hierarchically via an OrdinalEncoder, and passes numeric arrays through untouched via remainder='passthrough'.
3. **src/prediction.py**: Loads binary file models, wraps clean transformations, outputs prediction flags, and evaluates custom rule-based strategic retention insights to display directly on the dashboard page.

---

## 10. Tools & Technologies

* Core Engine: Python, Pandas, NumPy
* Visualizations: Matplotlib, Seaborn
* Modeling Logic: Scikit-Learn (Pipelines, ColumnTransformer, Models)
* Advanced Algos: XGBoost, Support Vector Machine (SVC)
* Serialization: Joblib
* Deployment System: Streamlit Engine Frameworks

---

## 11. How to Run Locally

1. **Clone your repository:**
   ```bash
   git clone https://github.com
   cd customer-churn-prediction
   ```

2. **Install dependencies listed inside requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Boot your application server:**
   ```bash
   python -m streamlit run app.py
   ```
