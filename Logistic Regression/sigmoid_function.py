import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Generate Synthetic Binary Classification Dataset

n_samples = 200

feature_1 = np.random.normal(5, 2, n_samples)
feature_2 = np.random.normal(3, 1.5, n_samples)

decision = 1.5 * feature_1 - 2 * feature_2 + 1
target = (decision > 5).astype(int)

dataset = pd.DataFrame(
    {
        "Feature1": feature_1,
        "Feature2": feature_2,
        "Target": target,
    }
)

X = dataset.drop(columns="Target")
y = dataset["Target"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression From Scratch


class LogisticRegressionSigmoid:
    """Binary Logistic Regression using Batch Gradient Descent."""

    def __init__(self, learning_rate=0.1, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def _sigmoid(self, X):
        """Compute sigmoid probabilities."""

        z = np.dot(X, self.weights)
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the model."""

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Add bias feature
        X = np.insert(X, 0, 1, axis=1)

        self.weights = np.zeros(X.shape[1])

        for _ in range(self.epochs):

            predictions = self._sigmoid(X)

            gradient = np.dot(X.T, (predictions - y))

            self.weights -= (
                self.learning_rate
                * gradient
                / X.shape[0]
            )

        return self

    def predict_proba(self, X):
        """Return predicted probabilities."""

        X = np.insert(X, 0, 1, axis=1)

        return self._sigmoid(X)

    def predict(self, X):
        """Return predicted class labels."""

        probabilities = self.predict_proba(X)

        return (probabilities >= 0.5).astype(int)


# Custom Logistic Regression

custom_model = LogisticRegressionSigmoid(
    learning_rate=0.001,
    epochs=90000,
)

custom_model.fit(X_train, y_train)

custom_predictions = custom_model.predict(X_test)

print(
    "Custom Logistic Regression Accuracy:",
    accuracy_score(y_test, custom_predictions),
)

# Scikit-Learn Logistic Regression

sklearn_model = LogisticRegression(
    
    solver="lbfgs",
    max_iter=5000,
)

sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)

print(
    "Scikit-Learn Logistic Regression Accuracy:",
    accuracy_score(y_test, sklearn_predictions),
)

# Compare Decision Boundaries

plt.figure(figsize=(7, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="black",
    alpha=0.8,
)

x_values = np.linspace(
    X_train[:, 0].min() - 1,
    X_train[:, 0].max() + 1,
    200,
)

# Custom model
bias = custom_model.weights[0]
w1 = custom_model.weights[1]
w2 = custom_model.weights[2]

plt.plot(
    x_values,
    -(bias + w1 * x_values) / w2,
    linewidth=2,
    label="Custom Logistic Regression",
)

# Scikit-Learn model
bias = sklearn_model.intercept_[0]
w1 = sklearn_model.coef_[0][0]
w2 = sklearn_model.coef_[0][1]

plt.plot(
    x_values,
    -(bias + w1 * x_values) / w2,
    "--",
    linewidth=2,
    label="Scikit-Learn Logistic Regression",
)

plt.title("Decision Boundary Comparison")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.grid(True)
plt.legend()

plt.show()