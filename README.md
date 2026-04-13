<<<<<<< HEAD
# Bank Marketing Classification Project

## Overview
This project is a machine learning pipeline that predicts whether a client will subscribe to a term deposit based on bank marketing data.

The workflow includes data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation using a Decision Tree classifier.

---

## Dataset
The dataset used is the Bank Marketing dataset, which contains information about clients contacted during a marketing campaign.

### Features include:
- Client demographics (age, job, marital status)
- Financial information (balance, loans, housing)
- Campaign-related attributes

### Target variable:
- `y`: whether the client subscribed to a term deposit (`yes` / `no`)

---

## Data Preprocessing

A custom `wrangle()` function is used for data cleaning and transformation:

### Steps:
- Convert categorical binary variables (`yes` / `no`) into boolean values
- Create a new feature `balance_class` based on balance distribution
- Create a feature `previous_bool` indicating previous contact
- Drop irrelevant or redundant columns

---

## Feature Engineering
- Balance values are grouped into categories (A–E) using quantiles
- Boolean encoding is applied to binary features
- Unnecessary columns are removed to reduce noise

---

## Model

A Decision Tree Classifier from scikit-learn is used.

### Workflow:
1. Encode categorical features using OrdinalEncoder
2. Split dataset into training and testing sets
3. Train a baseline Decision Tree model
4. Optimize hyperparameters using GridSearchCV

---

## Hyperparameter Tuning
The following parameters are optimized:
- max_depth
- criterion (gini, entropy)
- min_samples_split
- min_samples_leaf

Cross-validation (cv=10) is used for robust evaluation.

---

## Evaluation Metrics
The model is evaluated using:
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## Results
The final model is selected based on cross-validation performance and evaluated on the test set.

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
=======
# Bank-Marketing-Classification-Project
Classification model using Decision Tree and scikit-learn to predict customer term deposit subscription. Includes full data wrangling, feature engineering, and hyperparameter optimization.
>>>>>>> 0daba43c9bf5e9f751a9efad7dc448c7bfb1b6f6
