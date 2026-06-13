# Predicting Online News Popularity 📈

Machine learning project for predicting how many times an online article will be shared on social media using regression models and feature engineering.

The project explores the relationship between article characteristics and audience engagement, with the goal of estimating article popularity before it spreads.

---

## Overview

Online publishers constantly face a difficult question:

> *Will this article attract attention and generate engagement?*

This project uses metadata and content-related statistics from nearly 40,000 articles to predict the number of social shares an article will receive.

Instead of treating the problem as a binary classification task ("popular" vs. "not popular"), the objective is formulated as a regression problem, allowing us to estimate the expected number of shares directly.

The dataset contains 39,797 articles and 61 attributes collected from articles published on Mashable. :contentReference[oaicite:0]{index=0}

---

## Objectives

- Understand the factors associated with article popularity.
- Build interpretable regression models.
- Investigate the impact of feature engineering.
- Compare ordinary linear regression with regularized models.
- Analyze prediction errors and model limitations.
- Create a foundation for a deployable prediction system.

---

## Dataset

**Source:** UCI Machine Learning Repository

Dataset: **Online News Popularity**

- 39,797 samples
- 61 attributes
- Target variable:

```text
shares
```

Important predictors include:

- Article length
- Number of images and videos
- Keyword statistics
- Topic distributions
- Sentiment features
- Subjectivity scores
- Publishing weekday information

---

## Project Structure

```text
online-news-popularity/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
│
├── reports/
│   ├── findings.md
│   └── charts/
│
└── models/
```

---

## Workflow

```text
Exploratory Data Analysis
            ↓
Target Distribution Analysis
            ↓
Log Transformation of Shares
            ↓
Feature Engineering
            ↓
Train-Test Split
            ↓
Feature Scaling
            ↓
Linear Regression
            ↓
Ridge Regression
            ↓
Lasso Regression
            ↓
Model Evaluation
            ↓
Residual Analysis
```

---

## Feature Engineering

Additional features and transformations explored:

### Target Transformation

Because article shares are heavily skewed:

```python
y = np.log1p(shares)
```

### Derived Features

Examples include:

- Weekend indicator
- Title length
- Content length
- Number of media elements
- Keyword-related statistics
- Topic similarity scores

---

## Models

### Linear Regression

Baseline model used for interpretability and benchmarking.

### Ridge Regression

Introduces L2 regularization to reduce overfitting.

### Lasso Regression

Uses L1 regularization and performs implicit feature selection.

---

## Evaluation Metrics

Models are compared using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Key Questions

This project attempts to answer questions such as:

- Do longer articles receive more shares?
- Does sentiment influence popularity?
- Are weekend publications more successful?
- Which features contribute most to engagement?
- Can a simple linear model capture online popularity?

---

## Future Improvements

### Hyperparameter Tuning

- GridSearchCV
- Cross-validation

### Advanced Models

- Random Forest Regressor
- Gradient Boosting
- XGBoost
- LightGBM

### Explainability

- Feature importance analysis
- Residual diagnostics
- SHAP values

### Deployment

Build an interactive web application where users can input article statistics and obtain predicted popularity in real time.

Possible frameworks:

- Streamlit
- FastAPI
- Docker

---

## Results

*Results and visualizations will be added after experimentation.*

---

## References

Fernandes, K., Vinagre, P., Cortez, P., & Sernadela, P. (2015).

**Online News Popularity Dataset**

UCI Machine Learning Repository.

DOI:

```
10.24432/C5NS3V
```

---

## License

This project is released under the MIT License.