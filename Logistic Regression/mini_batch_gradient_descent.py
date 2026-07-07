import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# Generate Dataset


X, y = make_regression(
    n_samples=10000,
    n_features=2,
    noise=20,
    random_state=42
)

# Convert regression targets into binary labels
y = (y > np.median(y)).astype(int)

df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
df["Target"] = y



# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Feature Scaling


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Custom SGD Logistic Regression


class CustomSGDClassifier:

    def __init__(self, batch=100, lr=0.01, epoch=100):

        self.weights = None
        self.batch = batch
        self.lr = lr
        self.epoch = epoch

        self.loss_history = []

    def fit(self, x, y):

        x = np.asarray(x)
        y = np.asarray(y).ravel()

        x = np.insert(x, 0, 1, axis=1)

        self.weights = np.zeros(x.shape[1])

        for _ in range(self.epoch):

            indices = np.random.permutation(x.shape[0])

            for start in range(0, x.shape[0], self.batch):

                end = start + self.batch
                idx = indices[start:end]

                y_pred = self.sigmoid(x[idx] @ self.weights)

                error = y[idx] - y_pred

                grad = x[idx].T @ error / len(idx)

                self.weights += self.lr * grad

            # Binary Cross Entropy Loss
            pred = self.sigmoid(x @ self.weights)

            loss = -np.mean(
                y * np.log(pred + 1e-10)
                + (1 - y) * np.log(1 - pred + 1e-10)
            )

            self.loss_history.append(loss)

        return self

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def predict_proba(self, x):

        x = np.insert(x, 0, 1, axis=1)

        return self.sigmoid(x @ self.weights)

    def predict(self, x):

        probability = self.predict_proba(x)

        return (probability >= 0.5).astype(int)

    def score(self, x, y):

        prediction = self.predict(x)

        return accuracy_score(y, prediction)



# Train Custom Model


custom_model = CustomSGDClassifier(
    batch=64,
    lr=0.05,
    epoch=100,
)

custom_model.fit(X_train, y_train)

custom_accuracy = custom_model.score(X_test, y_test)

print(f"Custom SGD Accuracy : {custom_accuracy:.4f}")


# Train Sklearn Model


sklearn_model = LogisticRegression()

sklearn_model.fit(X_train, y_train)

prediction = sklearn_model.predict(X_test)

sklearn_accuracy = accuracy_score(y_test, prediction)

print(f"Sklearn Accuracy    : {sklearn_accuracy:.4f}")


# Plot Training Loss


plt.figure(figsize=(7, 5))

plt.plot(custom_model.loss_history)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")

plt.grid(True)

plt.show()



# Decision Boundary


x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]

prediction = custom_model.predict(grid)

prediction = prediction.reshape(xx.shape)

plt.figure(figsize=(7, 6))

plt.contourf(
    xx,
    yy,
    prediction,
    alpha=0.3,
)

plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolors="k",
)

plt.title("Decision Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()





# Decision Boundary Comparison (Zoomed)


plt.figure(figsize=(8, 6))

# Plot training points
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap="bwr",
    alpha=0.6,
    edgecolors="k",
    s=25,
)

# X values for drawing lines
x_values = np.linspace(-1.5, 1.5, 300)


# Custom SGD Line

w0, w1, w2 = custom_model.weights

y_custom = -(w0 + w1 * x_values) / w2

plt.plot(
    x_values,
    y_custom,
    color="red",
    linewidth=3,
    label="Custom SGD",
)


# Sklearn Logistic Regression Line

intercept = sklearn_model.intercept_[0]
coef1, coef2 = sklearn_model.coef_[0]

y_sklearn = -(intercept + coef1 * x_values) / coef2

plt.plot(
    x_values,
    y_sklearn,
    color="blue",
    linestyle="--",
    linewidth=3,
    label="Sklearn LogisticRegression",
)


# Zoom Around the Decision Boundary

plt.xlim(-1.5, 1.5)
plt.ylim(-5, 5)

plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.title("Decision Boundary Comparison (Zoomed)", fontsize=14)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()