# Logistic-Regression-Evaluation-Overview

# Airline Customer Satisfaction Prediction using Logistic Regression

## ✈️ Project Overview
This project analyzes passenger survey data from **Invistico Airline** using Python and Scikit-learn to build a binomial Logistic Regression classification model. The primary goal is to identify the critical drivers behind passenger satisfaction and translate these statistical insights into actionable business strategies for maximizing customer retention.

---

## 📊 Dataset & Features
The dataset consists of **129,880 rows** and **22 columns**, tracking customer demographics, flight details, and 14 specific in-flight service parameters rated on a 0–5 satisfaction scale.

### Preprocessing & Feature Engineering Choices:
* **Missing Value Imputation:** The `Arrival Delay in Minutes` feature contained 393 missing records. These values were imputed using the feature's **median** to prevent outliers from skewing the data.
* **Target Encoding:** The `satisfaction` label was mapped to binary integers: `1` for *satisfied* and `0` for *dissatisfied*.
* **Categorical Encoding:** Binary predictors (`Customer Type`, `Type of Travel`) were mapped to $0/1$ flags. The nominal multi-class feature `Class` was handled via one-hot encoding, dropping the first indicator (`Class_Business`) to prevent multi-collinearity issues.
* **Feature Scaling:** Standard scaling (`StandardScaler`) was applied to all predictors, transforming them to a mean of 0 and a standard deviation of 1. This ensures that the scale of a metric (e.g., Flight Distance vs. a 1-5 survey scale) does not distort coefficient magnitude comparisons.

---

## 📈 Model Performance Analytics
The data was split using a stratified **80/20 train/test configuration** to maintain a perfect representation of both customer states across evaluation sets. 

The binomial Logistic Regression model achieved a **baseline classification accuracy of 83%** on the test split.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.81      0.81      0.81     11759
           1       0.84      0.84      0.84     14217

    accuracy                           0.83     25976
