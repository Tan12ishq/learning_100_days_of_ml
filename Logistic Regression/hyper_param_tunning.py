import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

np.random.seed(42)


# Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target


# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Feature Scaling

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Logistic Regression using Mini Batch Gradient Descent

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Softmax Logistic Regression implemented
    from scratch with a sklearn-like API.
    """

    def __init__(
        self,
        learning_rate=0.1,
        max_iter=1000,
        batch_size=32,
        fit_intercept=True,
        random_state=None,
        tol=1e-4,
        l2_lambda=0.01
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.tol = tol
        self.l2_lambda=l2_lambda
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.n_iter_ = None

        self.encoder_ = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )

        self.loss_history_ = []

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")

        if len(X) != len(y):
            raise ValueError("X and y have different lengths.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        y_one_hot = self.encoder_.fit_transform(
            y.reshape(-1, 1)
        )

        if self.fit_intercept:
            X = np.insert(
                X,
                0,
                1,
                axis=1,
            )

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y_one_hot.shape[1]

        weights = (
            np.random.randn(
                n_features,
                n_classes,
            )
            * 0.01
        )

        self.loss_history_ = []

        for epoch in range(self.max_iter):

            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                scores = X_batch @ weights

                probabilities = self.softmax(scores)

                gradient = (X_batch.T @ (probabilities - y_batch)) / len(X_batch)

                if self.fit_intercept:
                    gradient[1:] += self.l2_lambda * weights[1:]
                else:
                    gradient += self.l2_lambda * weights

                weights -= (
                    self.learning_rate * gradient
                )

            probabilities = self.softmax(
                X @ weights
            )

            loss = -np.mean(
                np.sum(
                    y_one_hot
                    * np.log(probabilities + 1e-15),
                    axis=1,
                )
            )

            self.loss_history_.append(loss)

            if epoch > 0:

                if abs(
                    self.loss_history_[-2]
                    - self.loss_history_[-1]
                ) < self.tol:

                    self.n_iter_ = epoch + 1
                    break

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:].T

        else:
            self.intercept_ = np.zeros(
                n_classes
            )
            self.coef_ = weights.T

        if self.n_iter_ is None:
            self.n_iter_ = self.max_iter

        return self
    
    
    def predict_proba(self, X):

        return self.softmax(
            self.decision_function(X)
        )

    def predict_log_proba(self, X):

        probabilities = self.predict_proba(X)

        return np.log(probabilities + 1e-15)

    def predict(self, X):

        if self.coef_ is None:
            raise ValueError(
                "This LogisticRegression instance is not fitted yet."
            )

        probabilities = self.predict_proba(X)

        class_indices = np.argmax(
            probabilities,
            axis=1,
        )

        return self.classes_[class_indices]

    def decision_function(self, X):

        X = np.asarray(X)

        if self.coef_ is None:
            raise ValueError(
                "This LogisticRegression instance is not fitted yet."
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of features does not match training data."
            )

        if self.fit_intercept:

            X = np.insert(
                X,
                0,
                1,
                axis=1,
            )

            weights = np.vstack(
                (
                    self.intercept_,
                    self.coef_.T,
                )
            )

        else:
            weights = self.coef_.T

        return X @ weights

    def score(self, X, y):

        predictions = self.predict(X)

        return accuracy_score(
            y,
            predictions,
        )

    def softmax(self, z):

        z = z - np.max(
            z,
            axis=1,
            keepdims=True,
        )

        exp_values = np.exp(z)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )

    def evaluate(self, X, y):

        if self.coef_ is None:
            raise ValueError(
                "This LogisticRegression instance is not fitted yet."
            )

        predictions = self.predict(X)

        return {
            "Accuracy": accuracy_score(
                y,
                predictions,
            ),
            "Precision": precision_score(
                y,
                predictions,
                average="weighted",
            ),
            "Recall": recall_score(
                y,
                predictions,
                average="weighted",
            ),
            "F1 Score": f1_score(
                y,
                predictions,
                average="weighted",
            ),
        }
        
        



# Train Models

custom_logistic = LogisticRegression(
    learning_rate=0.1,
    max_iter=1000,
    batch_size=32,
    random_state=42,
)

custom_logistic.fit(x_train, y_train)

sklearn_logistic = SklearnLogisticRegression(
    random_state=42,
    max_iter=1000,
)

sklearn_logistic.fit(x_train, y_train)


# Compare Models

print("Custom Logistic  :", custom_logistic.score(x_test, y_test))
print("Sklearn Logistic :", sklearn_logistic.score(x_test, y_test))


# Hyperparameter Tuning
param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_iter": [10,100,300],
    "batch_size": [8, 16, 32],
    "l2_lambda": [0, 1e-2, 1e-1],
}
grid = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

grid.fit(x_train, y_train)


# Best Model

print("Best Parameters :", grid.best_params_)
print("Best CV Score   :", grid.best_score_)

best_model = grid.best_estimator_

print("Best Model Test Accuracy :", best_model.score(x_test, y_test))

print()

results = best_model.evaluate(x_test, y_test)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
    
    
    
# Graphs




# Training Loss

plt.figure(figsize=(6, 4))
plt.plot(custom_logistic.loss_history_)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.grid(True)
plt.show()


# Accuracy Comparison

accuracies = [
    custom_logistic.score(x_test, y_test),
    sklearn_logistic.score(x_test, y_test),
    best_model.score(x_test, y_test),
]

labels = [
    "Custom",
    "Sklearn",
    "tuned model",
]

plt.figure(figsize=(6, 4))
plt.bar(labels, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)

for i, value in enumerate(accuracies):
    plt.text(i, value + 0.002, f"{value:.3f}", ha="center")

plt.show()



# Confusion Matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_model.predict(x_test),
)

plt.title("Confusion Matrix")
plt.show()


# Precision, Recall and F1 Score

metrics = best_model.evaluate(x_test, y_test)

metric_names = [
    "Precision",
    "Recall",
    "F1 Score",
]

metric_values = [
    metrics["Precision"],
    metrics["Recall"],
    metrics["F1 Score"],
]

plt.figure(figsize=(6, 4))
plt.bar(metric_names, metric_values)

plt.title("Classification Metrics")
plt.ylabel("Score")
plt.ylim(0.9, 1.0)

for i, value in enumerate(metric_values):
    plt.text(i, value + 0.002, f"{value:.3f}", ha="center")

plt.show()


