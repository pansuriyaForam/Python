# 🚗 Car Price Prediction using Machine Learning

## 📌 Overview
This project predicts the resale price of used cars using vehicle attributes such as age, mileage, fuel type, horsepower, and additional features.

The goal is to build a regression model that provides accurate price estimates to support better decision-making for buyers and sellers.

---

## 🎯 Objectives

- Understand factors influencing used car prices  
- Perform exploratory data analysis (EDA)  
- Build and evaluate regression models  
- Interpret feature impact on pricing  

---

## 📂 Dataset

**Toyota Corolla Dataset**

Includes:

- Vehicle age  
- Mileage (KM driven)  
- Fuel type  
- Horsepower  
- Transmission type  
- Additional features  

**Target Variable:** `Price`

---

## 🔎 Workflow

### 1️⃣ Data Understanding
- Checked structure, data types, and summary statistics
- Verified absence of missing values

### 2️⃣ Exploratory Data Analysis
Key insights:

- Vehicle price decreases with age and mileage
- Higher horsepower correlates with higher price
- Automatic transmission and premium features increase value
- Price distribution is right-skewed

### 3️⃣ Data Preprocessing
- Removed irrelevant columns
- Encoded categorical variables
- Prepared features for modeling

### 4️⃣ Models Evaluated
- Linear Regression  
- Ridge Regression  
- Lasso Regression  

Ridge regression was selected for better generalization and coefficient stability.

---

## 📊 Model Performance

| Model | R² Score | RMSE |
|------|--------|------|
| Linear Regression | ~0.87 | ~1300 |
| Ridge Regression | **~0.88** | **~1250** |
| Lasso Regression | ~0.87 | ~1320 |

**Best Model:** Ridge Regression  
**Explained Variance:** ~88%

---

## 📈 Model Performance Visualization

### Actual vs Predicted Prices

This plot compares predicted prices with actual values.

```python
plt.scatter(y_test, y_pred_ridge)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()
```

**Interpretation:**
Points close to the diagonal line indicate accurate predictions.
---
### Residual Plot
```python
residuals = y_test - y_pred_ridge

plt.scatter(y_pred_ridge, residuals)
plt.axhline(y=0, linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```

**Interpretation:**
Random distribution around zero indicates good model fit.
--- 
### Error Distribution
```python
sns.histplot(residuals, bins=30)
plt.title("Error Distribution")
plt.show()
```
**Interpretation:**
Errors centered around zero suggest unbiased predictions.
---

### 🔍 Feature Impact
Top drivers influencing car price:
**Positive Impact**
- Horsepower
- Automatic transmission
- Premium features

**Negative Impact**
- Vehicle age
- Mileage
- Older manufacturing year

--- 
### 🧠 Key Insights

- Depreciation is primarily driven by age and mileage
- Feature enhancements significantly influence resale value
- Ridge regression improves stability when predictors are correlated
---

### ⚙️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
---

## 🚀 How to Run
``` Bash
    pip install pandas numpy matplotlib seaborn scikit-learn
```
```python
    python car_price_prediction.py
```
---
## ✅ Conclusion

This project demonstrates a complete machine learning workflow:

-  data analysis
- feature engineering
- regression modeling
- performance evaluation
- model interpretation

The final model explains most price variation and provides reliable price estimates.

