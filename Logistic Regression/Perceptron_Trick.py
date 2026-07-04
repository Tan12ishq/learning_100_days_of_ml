import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(42)

# Perceptron Classifier


class Perceptron:
    """
    Binary Perceptron classifier trained using stochastic updates.
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        """Train the perceptron."""

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Add bias column
        X = np.insert(X, 0, 1, axis=1)

        self.weights = np.zeros(X.shape[1])

        for _ in range(self.epochs):

            idx = np.random.randint(X.shape[0])

            prediction = int(np.dot(self.weights, X[idx]) >= 0)
            error = y[idx] - prediction

            self.weights += self.lr * error * X[idx]

        return self

    def predict(self, X):
        """Predict binary class labels."""

        X = np.insert(X, 0, 1, axis=1)

        scores = np.dot(X, self.weights)

        return (scores >= 0).astype(int)

# Generate a linearly separable binary classification dataset


n_samples = 200

feature1 = np.random.normal(5, 2, n_samples)
feature2 = np.random.normal(3, 1.5, n_samples)

decision = 1.5 * feature1 - 2 * feature2 + 1
target = (decision > 5).astype(int)

df = pd.DataFrame(
    {
        "Feature1": feature1,
        "Feature2": feature2,
        "Target": target,
    }
)

X = df.drop(columns="Target")
y = df["Target"]


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




# Custom Perceptron


perceptron = Perceptron(lr=0.1, epochs=2000)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

print("Perceptron Accuracy :", accuracy_score(y_test, y_pred))



# Scikit-Learn Logistic Regression


log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred_sk = log_reg.predict(X_test)

print("Logistic Regression Accuracy :", accuracy_score(y_test, y_pred_sk))



# ============================================================
# 1. Original Dataset
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(
    df[df["Target"] == 0]["Feature1"],
    df[df["Target"] == 0]["Feature2"],
    color="royalblue",
    label="Class 0",
    alpha=0.7,
)

plt.scatter(
    df[df["Target"] == 1]["Feature1"],
    df[df["Target"] == 1]["Feature2"],
    color="crimson",
    label="Class 1",
    alpha=0.7,
)

plt.title("Original Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 2. Perceptron Decision Boundary
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="black",
)

x_vals = np.linspace(
    X_train[:, 0].min() - 1,
    X_train[:, 0].max() + 1,
    200,
)

bias = perceptron.weights[0]
w1 = perceptron.weights[1]
w2 = perceptron.weights[2]

y_vals = -(bias + w1 * x_vals) / w2

plt.plot(
    x_vals,
    y_vals,
    color="black",
    linewidth=2,
    label="Decision Boundary",
)

plt.title("Perceptron Decision Boundary")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 3. Logistic Regression Decision Boundary
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="black",
)

bias = log_reg.intercept_[0]
w1 = log_reg.coef_[0][0]
w2 = log_reg.coef_[0][1]

y_vals = -(bias + w1 * x_vals) / w2

plt.plot(
    x_vals,
    y_vals,
    color="green",
    linewidth=2,
    label="Decision Boundary",
)

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 4. Compare Both Decision Boundaries
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="black",
    alpha=0.8,
)

# Perceptron
bias = perceptron.weights[0]
w1 = perceptron.weights[1]
w2 = perceptron.weights[2]

plt.plot(
    x_vals,
    -(bias + w1 * x_vals) / w2,
    linewidth=2,
    label="Perceptron",
)

# Logistic Regression
bias = log_reg.intercept_[0]
w1 = log_reg.coef_[0][0]
w2 = log_reg.coef_[0][1]

plt.plot(
    x_vals,
    -(bias + w1 * x_vals) / w2,
    "--",
    linewidth=2,
    label="Logistic Regression",
)

plt.title("Decision Boundary Comparison")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.grid(True)

plt.show()

