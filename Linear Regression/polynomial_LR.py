import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

np.random.seed(42)


# Load Dataset

df = pd.read_csv("../datasets/BudgetItaly.csv")

df.drop(
    ["year", "whouse", "pmisc", "pfood", "size", "pct", "rownames"],
    axis=1,
    inplace=True,
)

X = df.drop("totexp", axis=1)
y = df["totexp"]


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


# Linear Regression

lr = LinearRegression()
lr.fit(x_train, y_train)

print("Linear Regression")
print("Intercept :", lr.intercept_)
print("Coefficients :", lr.coef_)
print("R2 Score :", r2_score(y_test, lr.predict(x_test)))


# Polynomial Regression (Degree = 2)

pf = PolynomialFeatures(degree=2)

x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.transform(x_test)

lr = LinearRegression()
lr.fit(x_train_poly, y_train)

print("\nPolynomial Regression (Degree = 2)")
print("Intercept :", lr.intercept_)
print("Coefficients :", lr.coef_)
print("R2 Score :", r2_score(y_test, lr.predict(x_test_poly)))


# Best Degree using GridSearchCV

pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("lr", LinearRegression())
])

params = {
    "poly__degree": [1, 2, 3, 4, 5, 6, 7]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    cv=10,
    scoring="r2",
)

grid.fit(x_train, y_train)

print("\nGrid Search")
print("Best Degree :", grid.best_params_["poly__degree"])
print("Best CV Score :", grid.best_score_)
print("Test R2 :", grid.score(x_test, y_test))


