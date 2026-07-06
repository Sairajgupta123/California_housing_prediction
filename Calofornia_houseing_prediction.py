# ============================================================
# House Price Prediction using Machine Learning in Python
# Models Used:
# 1. Linear Regression
# 2. Polynomial Regression
# 3. Ridge Regression
# 4. Lasso Regression
# ============================================================

# -----------------------------
# Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from sklearn.datasets import fetch_california_housing

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Step 1: Load California Housing Dataset
# ============================================================

# Load dataset
data = fetch_california_housing()

# Print dataset description
print(data.DESCR)

# Convert dataset into a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column (House Price)
df["MedHouseVal"] = data.target

# Display first five records
print(df.head())


# ============================================================
# Step 2: Data Preprocessing
# ============================================================

# Separate Features (X) and Target Variable (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling (Standardization)
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using same scaler
X_test_scaled = scaler.transform(X_test)


# ============================================================
# Step 3: Linear Regression
# ============================================================

# Create Linear Regression model
lr = LinearRegression()

# Train model
lr.fit(X_train_scaled, y_train)

# Predict house prices
y_pred_lr = lr.predict(X_test_scaled)

# Evaluate model
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression")
print("RMSE :", rmse_lr)
print("R² Score :", r2_lr)


# ============================================================
# Step 4: Polynomial Regression
# ============================================================

# Convert features into Polynomial Features (Degree = 2)
poly = PolynomialFeatures(degree=2)

# Transform training and testing data
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train Linear Regression on Polynomial Features
pr = LinearRegression()
pr.fit(X_train_poly, y_train)

# Predict
y_pred_pr = pr.predict(X_test_poly)

# Evaluate
rmse_pr = np.sqrt(mean_squared_error(y_test, y_pred_pr))
r2_pr = r2_score(y_test, y_pred_pr)

print("\nPolynomial Regression")
print("RMSE :", rmse_pr)
print("R² Score :", r2_pr)


# ============================================================
# Step 5: Ridge Regression
# ============================================================

# Create Ridge Regression model
ridge = Ridge(alpha=1.0)

# Train model
ridge.fit(X_train_scaled, y_train)

# Predict
y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluate
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression")
print("RMSE :", rmse_ridge)
print("R² Score :", r2_ridge)


# ============================================================
# Step 6: Lasso Regression
# ============================================================

# Create Lasso Regression model
lasso = Lasso(alpha=1.0)

# Train model
lasso.fit(X_train_scaled, y_train)

# Predict
y_pred_lasso = lasso.predict(X_test_scaled)

# Evaluate
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression")
print("RMSE :", rmse_lasso)
print("R² Score :", r2_lasso)


# ============================================================
# Step 7: Compare All Models
# ============================================================

# Create comparison table
result = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "RMSE": [
        rmse_lr,
        rmse_pr,
        rmse_ridge,
        rmse_lasso
    ],
    "R² Score": [
        r2_lr,
        r2_pr,
        r2_ridge,
        r2_lasso
    ]
})

print("\nModel Comparison")
print(result)


# ============================================================
# Step 8: Predict House Price for a New Sample
# ============================================================

# Select the last record from test dataset
test_sample = X_test.tail(1)

print("\nTest Sample")
print(test_sample)

# Scale the sample
test_sample_scaled = scaler.transform(test_sample)

# Convert into Polynomial Features
test_sample_poly = poly.transform(test_sample_scaled)

# Predict using all models
prediction_linear = lr.predict(test_sample_scaled)
prediction_poly = pr.predict(test_sample_poly)
prediction_ridge = ridge.predict(test_sample_scaled)
prediction_lasso = lasso.predict(test_sample_scaled)

# Display predictions
print("\nPredicted House Price")

print("Linear Regression      :", prediction_linear[0])
print("Polynomial Regression  :", prediction_poly[0])
print("Ridge Regression       :", prediction_ridge[0])
print("Lasso Regression       :", prediction_lasso[0])

# Actual House Price
print("\nActual House Price")
print(y_test.tail(1).values[0])


# ============================================================
# End of Program
# ============================================================
