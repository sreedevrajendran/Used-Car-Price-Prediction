readme_content = """# Used Car Price Prediction

Estimating the fair market value of used vehicles using regression-based machine learning algorithms and real-world datasets.

---

## 📌 Project Overview

### Objective
Develop a robust, data-driven machine learning model that accurately estimates the fair market resale value of used cars based on historical sales and technical data.

### Approach
Implemented a Supervised Regression methodology to extract patterns, minimizing error margins to deliver highly accurate pricing targets.

### Goal
Provide reliable price guidelines for both sellers and buyers to ensure fair automotive market transactions.

---

## 📊 Dataset Features

The dataset incorporates various core identifiers, usage metrics, and technical specifications that dictate a vehicle's market value:

* **Vehicle Info:** Contains core identifiers including Car Brand, Specific Model, Year of Manufacture, and Fuel Type (Petrol, Diesel, or CNG).
  * *Fuel Type Distribution:* Predominantly split between **Diesel (53%)** and **Petrol (46%)**, with **CNG** making up **1%**. The dataset represents the vast majority of historical resale data available for model training.
* **Usage Metrics:** Features key depreciation factors such as total Kilometers Driven, Ownership History (1st, 2nd, 3rd hand), and Location.
* **Technical Specs:** Includes mechanical capabilities detailing Engine Displacement (CC), Power Output (bhp), Mileage (kmpl), and Seating Capacity.

---

## ⚙️ Data Preparation & Workflow

1. **Data Cleaning:** Extracted numeric values from string-based columns by stripping units like `'kmpl'`, `'CC'`, and `'bhp'`, converting them to float formats.
2. **Handling Missing Values:** Imputed missing numerical variables (Engine, Power, Mileage) with median values to maintain distribution integrity.
3. **Feature Selection:** Isolated `'Price'` as the target variable, selecting `Year`, `Kilometers Driven`, `Mileage`, `Engine`, `Power`, and `Seats` as core predictors.
4. **Data Splitting:** Divided the dataset into an **80% training set** and a **20% testing set** to objectively evaluate the model's predictive capabilities.

---

## 🧠 Regression Methodology

### Linear Regression Modeling
Established as our core model to map the direct linear relationships between independent variables (e.g., manufacture year, kilometers driven) and the target price. This algorithm models the expected value of the car as a weighted sum of its features, allowing for straightforward, continuous price forecasting.

### Model Interpretability & Efficiency
* **High Interpretability:** Linear Regression provides a clear mathematical formula to understand the immediate financial impact of individual technical specifications.
* **Computational Efficiency:** The model requires less computational overhead, effectively limits overfitting on linearly correlated features, and maintains transparent decision-making logic.

---

## 📈 Model Evaluation & Performance

### Evaluation Metric
The model's accuracy was quantified using the R-squared ($R^2$) metric, imported directly via the Python `sklearn.metrics` module:

```python
r_squared = r2_score(y_test, y_pred) * 100
