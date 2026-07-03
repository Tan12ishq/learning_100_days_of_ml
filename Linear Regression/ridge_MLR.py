import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

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


# Ridge Regression using Mini Batch Gradient Descent

class RidgeRegularization(RegressorMixin, BaseEstimator):
    """Mini-batch Gradient Descent implementation of Ridge Regression."""

    def __init__(self, alpha=0.01, lr=0.01, epochs=100, batch=100):
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch = batch

    def fit(self, x, y):

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.ravel()

        n_samples, n_features = x.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for epoch in range(self.epochs):

            indices = np.random.permutation(n_samples)

            for j in range(int(n_samples / self.batch)):

                start = j * self.batch
                end = start + self.batch
                idx = indices[start:end]

                x_batch = x[idx]
                y_batch = y[idx]

                y_pred = np.dot(x_batch, self.coef_) + self.intercept_

                residue = y_batch - y_pred

                intercept_grad = -2 * np.mean(residue)
                coef_grad = (
                    -2 * np.dot(residue, x_batch) / self.batch
                    + 2 * self.alpha * self.coef_
                )

                self.coef_ -= self.lr * coef_grad
                self.intercept_ -= self.lr * intercept_grad

        return self

    def predict(self, x):

        return np.dot(x, self.coef_) + self.intercept_

    def score(self, x, y):

        return r2_score(y, self.predict(x))


# Train Models

custom_ridge = RidgeRegularization(alpha=0.1)
custom_ridge.fit(x_train, y_train)

sklearn_ridge = Ridge(alpha=0.01)
sklearn_ridge.fit(x_train, y_train)

linear = LinearRegression()
linear.fit(x_train, y_train)


# Compare Models

print("Custom Ridge      :", custom_ridge.score(x_test, y_test))
print("Sklearn Ridge     :", sklearn_ridge.score(x_test, y_test))
print("Linear Regression :", linear.score(x_test, y_test))


# Hyperparameter Tuning

param_grid = {
    "alpha": [0.001, 0.01,0.05],
    "lr": [0.001,0.05,0.005,0.01,0.1],
    "epochs": [15,25,50],
    "batch": [ 20, 32,64],
}

grid = GridSearchCV(
    estimator=RidgeRegularization(),
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
)

grid.fit(x_train, y_train)


# Best Model

print("Best Parameters :", grid.best_params_)
print("Best CV Score   :", grid.best_score_)

best_model = grid.best_estimator_

print("Best Model Test R²:", best_model.score(x_test, y_test))