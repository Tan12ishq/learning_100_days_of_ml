import random

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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


# Sklearn Linear Regression

lr = LinearRegression()
lr.fit(x_train, y_train)

print("\nSklearn Linear Regression (OLS)")
print("Intercept :", lr.intercept_)
print("Coefficients :", lr.coef_)
print("R2 Score :", r2_score(y_test, lr.predict(x_test)))


# Mini-Batch Gradient Descent From Scratch

class MBGDRegressor:

    """
    Linear Regression using Mini-Batch Gradient Descent.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=100,
        decay=1e-5,
        batch_size=100,
        sampling="random",
        lr_schedule="exponential"
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decay = decay
        self.batch_size = batch_size
        self.sampling = sampling
        self.lr_schedule = lr_schedule

        self.intercept_ = 0.0
        self.coef_ = None
        self.loss_history = []

    def get_lr(self, t):

        if self.lr_schedule == "constant":
            return self.learning_rate

        elif self.lr_schedule == "exponential":
            return self.learning_rate * np.exp(-self.decay * t)

        else:
            raise ValueError("Invalid learning rate schedule.")

    def fit(self, X, y):

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.ravel()

        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        t = 0

        for epoch in range(self.epochs):

            # Shuffle indices
            if self.sampling == "shuffle":
                indices = np.random.permutation(n_samples)

            for j in range(n_samples // self.batch_size):

                # Select batch
                if self.sampling == "random":
                    idx = random.sample(range(n_samples), self.batch_size)

                elif self.sampling == "shuffle":
                    start = j * self.batch_size
                    end = start + self.batch_size
                    idx = indices[start:end]

                else:
                    raise ValueError("Invalid sampling strategy.")

                X_batch = X[idx]
                y_batch = y[idx]

                # Learning rate
                lr = self.get_lr(t)
                t += 1

                # Prediction
                y_pred = np.dot(X_batch, self.coef_) + self.intercept_

                # Gradients
                intercept_grad = -2 * np.mean(y_batch - y_pred)
                coef_grad = -2 * np.dot((y_batch - y_pred), X_batch) / self.batch_size

                # Update
                self.intercept_ -= lr * intercept_grad
                self.coef_ -= lr * coef_grad

            # Epoch loss
            predictions = np.dot(X, self.coef_) + self.intercept_
            loss = np.mean((y - predictions) ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):

        return r2_score(y, self.predict(X))


# Train Custom MBGD

mbgd = MBGDRegressor(
    learning_rate=0.01,
    epochs=100,
    decay=1e-5,
    batch_size=100,
    sampling="shuffle"
)

mbgd.fit(x_train, y_train)

print("\nCustom MBGDRegressor")
print("Intercept :", mbgd.intercept_)
print("Coefficients :", mbgd.coef_)
print("R2 Score :", mbgd.score(x_test, y_test))


# Final Comparison

print("\nFinal Comparison")
print(f"LinearRegression R² : {r2_score(y_test, lr.predict(x_test)):.4f}")
print(f"Custom MBGD R²      : {mbgd.score(x_test, y_test):.4f}")


# Training Loss Plot

fig = px.line(
    x=range(1, len(mbgd.loss_history) + 1),
    y=mbgd.loss_history,
    markers=True,
    labels={
        "x": "Epoch",
        "y": "MSE Loss",
    },
    title="Training Loss vs Epoch",
)

fig.show()