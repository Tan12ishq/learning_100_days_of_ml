import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



# Load and Prepare Dataset


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

sklearn_pred = lr.predict(x_test)

print("========== Sklearn ==========")
print("Intercept :", lr.intercept_)
print("Coefficients :", lr.coef_)
print("R2 Score :", r2_score(y_test, sklearn_pred))



# Gradient Descent Regressor


class GDRegressor:

    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.intercept = 0.0
        self.coefficients = None

    def fit(self, X, y):

        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        y = y.ravel()

        n_samples, n_features = X.shape

        self.coefficients = np.zeros(n_features)

        for _ in range(self.epochs):

            # Predictions
            y_pred = X @ self.coefficients + self.intercept

            # Errors
            error = y - y_pred

            # Gradients
            db = 2 * np.mean(error)
            dw = (2 / n_samples) * (X.T @ error)

            # Parameter Update
            self.intercept += self.learning_rate * db
            self.coefficients += self.learning_rate * dw

    def predict(self, X):

        return X @ self.coefficients + self.intercept



# Train Gradient Descent Model


gd = GDRegressor(
    learning_rate=0.01,
    epochs=100000
)

gd.fit(x_train, y_train)

gd_pred = gd.predict(x_test)

print("\n========== Gradient Descent ==========")
print("Intercept :", gd.intercept)
print("Coefficients :", gd.coefficients)
print("R2 Score :", r2_score(y_test, gd_pred))