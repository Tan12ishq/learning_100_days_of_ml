import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

np.random.seed(42)


# Load Dataset

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
)


# Feature Scaling

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Logistic Regression

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(x_train, y_train)

print("Logistic Regression")
print("Accuracy :", accuracy_score(y_test, log_reg.predict(x_test)))


# Polynomial Logistic Regression (Degree = 2)

pf = PolynomialFeatures(degree=2)

x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.transform(x_test)

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(x_train_poly, y_train)

print("\nPolynomial Logistic Regression (Degree = 2)")
print("Accuracy :", accuracy_score(y_test, log_reg.predict(x_test_poly)))


# Best Degree using GridSearchCV

pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("lr", LogisticRegression(max_iter=1000))
])

params = {
    "poly__degree": [1, 2, 3, 4]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    cv=10,
    scoring="accuracy",
)

grid.fit(x_train, y_train)

print("\nGrid Search")
print("Best Degree :", grid.best_params_["poly__degree"])
print("Best CV Score :", grid.best_score_)
print("Test Accuracy :", grid.score(x_test, y_test))