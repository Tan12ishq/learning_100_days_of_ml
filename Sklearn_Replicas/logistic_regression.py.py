import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin


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
        tol=1e-4
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.tol = tol

        # Learned attributes
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
        """
        Train Softmax Logistic Regression using
        Mini-Batch Gradient Descent.
        """
        
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

        y_one_hot = self.encoder_.fit_transform(y.reshape(-1, 1))

        # Add Bias Column
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y_one_hot.shape[1]

        # Weight Matrix
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

                weights -= self.learning_rate * gradient

            # Compute Loss
            probabilities = self.softmax(X @ weights)

            loss = -np.mean(
                np.sum(
                    y_one_hot * np.log(probabilities + 1e-15),
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
                    

        # Save learned parameters
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:].T
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = weights.T

        if self.n_iter_ is None:
            self.n_iter_ = self.max_iter
        

        return self

    def predict_proba(self, X):
        return self.softmax(self.decision_function(X))

    def predict_log_proba(self, X):
        """
        Return log class probabilities.
        """

        probabilities = self.predict_proba(X)

        return np.log(probabilities + 1e-15)

    def predict(self, X):
        """
        Predict class labels.
        """
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
            X = np.insert(X, 0, 1, axis=1)
            weights = np.vstack((self.intercept_, self.coef_.T))
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
        """
        Compute numerically stable Softmax probabilities.
        """

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
        """
        Return common classification metrics.
        """
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
