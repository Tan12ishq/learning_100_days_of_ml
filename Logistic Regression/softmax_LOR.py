import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

plt.style.use("ggplot")

# Generate Synthetic Multi-Class Dataset

X, y = make_blobs(
    n_samples=10000,
    centers=4,
    n_features=2,
    cluster_std=2.0,
    random_state=42,
)

df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"],
)

df["Target"] = y

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Custom Softmax Logistic Regression

class CustomSoftmaxLogisticRegression:
    """
    Mini-Batch Softmax Logistic Regression
    implemented from scratch.
    """

    def __init__(
        self,
        learning_rate=0.1,
        epochs=500,
        batch_size=32,
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = None
        self.loss_history = []
        self.encoder = None

    def fit(self, X, y):
        """
        Train the model using Mini-Batch
        Gradient Descent.
        """

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # Add Bias Feature
        X = np.insert(X, 0, 1, axis=1)

        # One-Hot Encode Labels
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )

        y_one_hot = self.encoder.fit_transform(y)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y_one_hot.shape[1]

        self.weights = np.zeros(
            (n_features, n_classes)
        )

        for epoch in range(self.epochs):

            shuffled_indices = np.random.permutation(
                n_samples
            )

            X_shuffled = X[shuffled_indices]
            y_shuffled = y_one_hot[shuffled_indices]

            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                probabilities = self.softmax(
                    X_batch @ self.weights
                )

                error = probabilities - y_batch

                gradient = (
                    X_batch.T @ error
                ) / len(X_batch)

                self.weights -= (
                    self.learning_rate * gradient
                )

            probabilities = self.softmax(
                X @ self.weights
            )

            loss = -np.mean(
                np.sum(
                    y_one_hot
                    * np.log(probabilities + 1e-15),
                    axis=1,
                )
            )

            self.loss_history.append(loss)

        return self

    def softmax(self, z):
        """
        Compute Softmax probabilities.
        """

        z = z - np.max(
            z,
            axis=1,
            keepdims=True,
        )

        exp_values = np.exp(z)

        return (
            exp_values
            / np.sum(
                exp_values,
                axis=1,
                keepdims=True,
            )
        )

    def predict_proba(self, X):
        """
        Return class probabilities.
        """

        X = np.asarray(X)

        X = np.insert(
            X,
            0,
            1,
            axis=1,
        )

        return self.softmax(
            X @ self.weights
        )

    def predict(self, X):
        """
        Predict class labels.
        """

        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1,
        )

    def score(self, X, y):
        """
        Return model accuracy.
        """

        predictions = self.predict(X)

        return accuracy_score(
            y,
            predictions,
        )


# Train Custom Model

custom_model = CustomSoftmaxLogisticRegression(
    learning_rate=0.1,
    epochs=500,
    batch_size=32,
)

custom_model.fit(
    X_train,
    y_train,
)

custom_predictions = custom_model.predict(
    X_test
)

custom_accuracy = accuracy_score(
    y_test,
    custom_predictions,
)

# Train Sklearn Model

sklearn_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
)

sklearn_model.fit(
    X_train,
    y_train,
)

sklearn_predictions = sklearn_model.predict(
    X_test
)

sklearn_accuracy = accuracy_score(
    y_test,
    sklearn_predictions,
)

print("Model Performance")

print(
    f"Custom Softmax Accuracy : {custom_accuracy:.4f}"
)
print(
    f"Sklearn Accuracy        : {sklearn_accuracy:.4f}"
)
print()

# Mesh Grid (Used for all plots)

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1

y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]
# 1. Training Dataset

plt.figure(figsize=(7, 6))

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="viridis",
    edgecolors="black",
    s=35,
)

plt.title("Synthetic Multi-Class Training Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()


# 2. Cross-Entropy Loss

plt.figure(figsize=(7, 5))

plt.plot(
    custom_model.loss_history,
    linewidth=2,
)

plt.title("Cross-Entropy Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()


# 3. Custom Decision Boundary

custom_boundary = custom_model.predict(grid)
custom_boundary = custom_boundary.reshape(xx.shape)

plt.figure(figsize=(7, 6))

plt.contourf(
    xx,
    yy,
    custom_boundary,
    alpha=0.35,
    cmap="viridis",
)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="viridis",
    edgecolors="black",
    s=25,
)

plt.title("Custom Softmax Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()










# 6. Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    custom_predictions,
    cmap="Blues",
)

plt.title("Confusion Matrix - Custom Softmax")

plt.tight_layout()
plt.show()






