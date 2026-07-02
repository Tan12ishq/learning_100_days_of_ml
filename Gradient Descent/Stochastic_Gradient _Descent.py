import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor as SklearnSGDRegressor
from sklearn.metrics import r2_score

np.random.seed(42)


# Load Dataset


df = pd.read_csv("../datasets/BudgetItaly.csv")

df.drop(
    ['year', 'whouse', 'pmisc', 'pfood', 'size', 'pct', 'rownames'],
    axis=1,
    inplace=True
)

X = df.drop("totexp", axis=1)
y = df["totexp"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)


# Feature Scaling


scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Sklearn Linear Regression


lr = LinearRegression()

lr.fit(x_train, y_train)


print("\n Sklearn Linear Regression (OLS)")


print("Intercept :", lr.intercept_)
print("Coefficients :", lr.coef_)
print("R2 Score :", r2_score(y_test, lr.predict(x_test)))


# SGD From Scratch


class SGDRegressor:

    """
    Linear Regression using Stochastic Gradient Descent
    with Exponential Learning Rate Decay.
    """

    def __init__(self,
                 learning_rate=0.001,
                 epochs=50,
                 decay=1e-5):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decay = decay

        self.intercept_ = 0.0
        self.coef_ = None

        self.loss_history = []

    def get_lr(self, t):
        return self.learning_rate * np.exp(-self.decay * t)

    def fit(self, X, y):

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.ravel()

        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)

        for epoch in range(self.epochs):

            indices = np.random.permutation(n_samples)

            for j, idx in enumerate(indices):

                t = epoch * n_samples + j
                lr = self.get_lr(t)

                # Prediction
                y_pred = X[idx] @ self.coef_ + self.intercept_

                # Error
                error = y[idx] - y_pred

                # Gradients
                db = 2 * error
                dw = 2 * X[idx] * error

                # Update
                self.intercept_ += lr * db
                self.coef_ += lr * dw
            # Loss after every epoch
            predictions = X @ self.coef_ + self.intercept_
            loss = np.mean((y - predictions) ** 2)
            self.loss_history.append(loss)

            
            

    def predict(self, X):

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):

        return r2_score(y, self.predict(X))


# Train Custom SGD


sgd = SGDRegressor(
    learning_rate=0.001,
    epochs=50,
    decay=1e-5
)

sgd.fit(x_train, y_train)


print("\n Custom SGDRegressor")


print("Intercept :", sgd.intercept_)
print("Coefficients :", sgd.coef_)
print("R2 Score :", sgd.score(x_test, y_test))


# Sklearn SGDRegressor


sk_sgd = SklearnSGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.001,
    max_iter=50,
    shuffle=True,
    random_state=42,
    tol=None
)

sk_sgd.fit(x_train, y_train)


print("\n Sklearn SGDRegressor")


print("Intercept :", sk_sgd.intercept_[0])
print("Coefficients :", sk_sgd.coef_)
print("R2 Score :", r2_score(y_test, sk_sgd.predict(x_test)))


# Final Comparison



print("\n Final Comparison")


print(f"LinearRegression R² : {r2_score(y_test, lr.predict(x_test)):.4f}")
print(f"Custom SGD R²       : {sgd.score(x_test, y_test):.4f}")
print(f"Sklearn SGD R²      : {r2_score(y_test, sk_sgd.predict(x_test)):.4f}")



fig = px.line(
    x=range(1, len(sgd.loss_history) + 1),
    y=sgd.loss_history,
    markers=True,
    labels={
        "x": "Epoch",
        "y": "MSE Loss"
    },
    title="Training Loss vs Epoch"
)

fig.show()