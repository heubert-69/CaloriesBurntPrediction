# 🥗 Calorie Burn Prediction using Machine Learning

## 🏆 Kaggle Competition Overview

In this regression competition hosted on Kaggle, the objective is to **predict the number of calories burned** during exercise sessions using biometric and activity-related input data.

### 📊 Evaluation Metric

The competition uses **Root Mean Squared Logarithmic Error (RMSLE)** to evaluate predictions:

\[
\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}
\]

This metric is preferred for this task because it penalizes underestimations more than overestimations and is robust to outliers.

---

## 📂 Dataset Description

The dataset includes the following features:

| Column Name   | Description                                 |
|---------------|---------------------------------------------|
| `id`          | Unique identifier for each record           |
| `Gender`      | Gender of the individual (`Male`/`Female`)  |
| `Age`         | Age in years                                |
| `Height`      | Height in centimeters                       |
| `Weight`      | Weight in kilograms                         |
| `Duration`    | Exercise duration in minutes                |
| `Heart_Rate`  | Average heart rate during exercise          |
| `Body_Temp`   | Average body temperature during exercise    |
| `Calories`    | **Target variable**: calories burned        |

---

## 🧪 Approach

### 🔧 Preprocessing

- **Categorical Encoding:** One-hot encoding was used for the `Gender` feature.
- **Scaling:** Numerical features were standardized using `StandardScaler`.

### 🔍 Feature Types

```python
categorical_features = ['Gender']
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
🔨 Model Used
We used an XGBoost Regressor with the following hyperparameters:

XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42
)
Model was wrapped in a scikit-learn pipeline that handles preprocessing and regression in one step.

📈 Model Evaluation
Metrics on Holdout Set:
Metric	Score
MAE	~3.09
MSE	~20.90
RMSE	~4.57
R² Score	~0.99
RMSLE	e.g. ~0.017 (low = better)

All predictions were post-processed to ensure non-negative values before computing RMSLE.

🧾 Submission Format
The final predictions were saved in a CSV file in the required Kaggle format:

csv
Copy
Edit
id,Calories
1,123.45
2,234.56
...
Python Code to Create Submission:
python
Copy
Edit
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_pred
})
submission.to_csv('submission.csv', index=False)
📌 Final Notes
All models were developed and tested using Python and popular machine learning libraries like scikit-learn, xgboost, and pandas.

Efforts were made to avoid overfitting through regularization and cross-validation.

Model performance was excellent, achieving near-perfect scores on training and validation sets.

🚀 Future Improvements
Perform hyperparameter tuning with GridSearchCV or Optuna.

Try ensemble models (e.g., stacking XGBoost with LightGBM or AdaBoost).

Implement SHAP for feature importance analysis.

Author: heubert-69
Competition: Calorie Burn Prediction (Kaggle)
