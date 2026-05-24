# MPG Prediction — Multiple Linear Regression (mtcars)

> Predicting fuel efficiency from a car's specs — and being deliberate about which features earn their place in the model.

An end-to-end regression project on the classic **mtcars** dataset that predicts miles-per-gallon (`mpg`) from vehicle characteristics. Goes beyond fitting a line: it handles outliers, diagnoses multicollinearity, and validates the model against both train and test sets.

---

## 📌 What This Project Does

The goal is a clean, well-reasoned linear model — not just a high R². The pipeline cleans the data, makes evidence-based feature decisions, trains a Multiple Linear Regression model, and checks for overfitting by comparing train vs test performance.

---

## 🔬 Pipeline

**1. Data Inspection**
- Missing-value and dtype checks across all columns

**2. Outlier Handling**
- Boxplots for all continuous features (`mpg`, `disp`, `hp`, `drat`, `wt`, `qsec`)
- Outliers imputed using the **IQR method** — capped at the upper bound (Q3 + 1.5·IQR) rather than dropped, preserving the small sample size

**3. Feature Selection — with reasoning**
- Pairplots + correlation heatmap to map relationships
- **`disp` dropped** due to multicollinearity with `cyl`
- `qsec` excluded to keep the model focused on the strongest predictors

**4. Modeling**
- 80/20 train-test split, `StandardScaler` on features
- `LinearRegression` from scikit-learn

**5. Evaluation — train vs test**
- MSE, RMSE, MAE, and R² computed on **both** sets to detect overfitting
- Side-by-side R² comparison

**6. Model Persistence**
- Final model serialized with `joblib` (`mtcars_regression_model.pkl`)

---

## 🛠 Stack

`Python` `scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `joblib`

**Techniques:** IQR Outlier Imputation · Multicollinearity Analysis · Feature Scaling · Multiple Linear Regression · Train/Test Validation

---

## 🗂 Repository Structure

```
Regression_Model_Building_mtcars.ipynb   → Full pipeline notebook
mtcars_xlsx_-_mtcars.csv                 → Dataset
mtcars_regression_model.pkl              → Saved trained model
```

---

## 🚀 Run It

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
jupyter notebook Regression_Model_Building_mtcars.ipynb
```

---

## 📬 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shenbaga-arun/)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat&logo=todoist&logoColor=white)](https://shenbagaarunk.github.io/)
