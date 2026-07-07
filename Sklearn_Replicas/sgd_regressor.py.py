"""
SGDRegressor implemented from scratch using CuPy.

Features
--------
- Linear Regression
- Stochastic / Mini-Batch Gradient Descent
- L1, L2 and Elastic Net Regularization
- Multiple Learning Rate Schedules
- Early Stopping
- sklearn-compatible API
"""

import cupy as cp

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class SGDReg(BaseEstimator, RegressorMixin):
    """
    Linear Regression trained using Stochastic Gradient Descent.

    Parameters
    ----------
    learning_rate : {"constant", "optimal", "invscaling", "exponential"}
        Strategy used to update the learning rate.

    penalty : {"l1", "l2", "elasticnet", None}
        Regularization applied during training.

    batch_size : int
        Number of samples processed in one update.

    fit_intercept : bool
        Whether to learn an intercept term.
    """

    def __init__(
        self,
        learning_rate="invscaling",
        max_iter=1000,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        batch_size=1,
        shuffle=True,
        tol=1e-3,
        eta0=0.01,
        decay=1e-3,
        t0=1,
        random_state=42,
        power_t=0.25,
        n_iter_no_change=5,
        fit_intercept=True,
    ):

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tol = tol
        self.eta0 = eta0
        self.decay = decay
        self.t0 = t0
        self.power_t = power_t
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change
        self.fit_intercept = fit_intercept

        # Model parameters
        self.coef_ = None
        self.intercept_ = None

        # Training information
        self.loss_curve_ = []
        self.n_iter_ = None

        # Public API
    
    def fit(self, X, y):
        """
        Train the model using SGD.
        """

        X, y = self._preprocess(X, y)

        n_samples, n_features = X.shape

        self.coef_ = cp.zeros(n_features, dtype=X.dtype)
        self.intercept_ = 0.0

        best_loss = cp.inf
        no_improvement = 0
        t = 1

        for epoch in range(self.max_iter):

            if self.shuffle:
                indices = cp.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):

                lr = self._get_learning_rate(t)

                idx = self._get_batch_indices(start, indices)

                X_batch = X[idx]
                y_batch = y[idx]

                grad_coef, grad_intercept = self._compute_gradient(
                    X_batch,
                    y_batch,
                )

                self.coef_ -= lr * grad_coef

                if self.fit_intercept:
                    self.intercept_ -= lr * grad_intercept

                t += 1

            loss = float(self._compute_loss(X, y))
            self.loss_curve_.append(loss)

            best_loss, no_improvement, stop = self._should_stop(
                loss,
                best_loss,
                no_improvement,
            )

            if stop:
                self.n_iter_ = epoch + 1
                break

        if self.n_iter_ is None:
            self.n_iter_ = self.max_iter

        return self

    def predict(self, X):
        """
        Predict target values.
        """

        X = cp.asarray(X, dtype=cp.float64)

        pred = X @ self.coef_

        if self.fit_intercept:
            pred += self.intercept_

        return pred

    def score(self, X, y):
        """
        Return R² score.
        """

        y_pred = self.predict(X)

        return r2_score(
            cp.asnumpy(y),
            cp.asnumpy(y_pred),
        )

        # Training Helpers
    
    def _preprocess(self, X, y):
        """
        Validate parameters and convert data to CuPy arrays.
        """

        self._validate_params()

        cp.random.seed(self.random_state)

        X = cp.asarray(X, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.float64).ravel()

        return X, y

    def _get_batch_indices(self, start, indices):
        """
        Return indices for the current mini-batch.
        """

        end = start + self.batch_size

        if self.shuffle:
            return indices[start:end]

        return slice(start, end)

    def _compute_gradient(self, X_batch, y_batch):
        """
        Compute gradient for one mini-batch.
        """

        scale = 2.0 / len(y_batch)

        y_pred = self.predict(X_batch)

        residual = y_pred - y_batch

        grad_coef = scale * X_batch.T @ residual
        grad_intercept = scale * cp.sum(residual)

        grad_coef = self._compute_regularization_gradient(
            grad_coef
        )

        return grad_coef, grad_intercept

    def _compute_loss(self, X, y):
        """
        Compute objective function.
        """

        y_pred = self.predict(X)

        loss = cp.mean((y - y_pred) ** 2)

        if self.penalty == "l2":

            loss += self.alpha * cp.sum(self.coef_ ** 2)

        elif self.penalty == "l1":

            loss += self.alpha * cp.sum(cp.abs(self.coef_))

        elif self.penalty == "elasticnet":

            loss += self.alpha * (
                self.l1_ratio * cp.sum(cp.abs(self.coef_))
                + (1 - self.l1_ratio) * cp.sum(self.coef_ ** 2)
            )

        return loss

    def _should_stop(
        self,
        loss,
        best_loss,
        count,
    ):
        """
        Check early stopping criterion.
        """

        if best_loss - loss > self.tol:
            return loss, 0, False

        count += 1

        return (
            best_loss,
            count,
            count >= self.n_iter_no_change,
        )

        # Optimization Helpers
    
    def _compute_regularization_gradient(self, grad_coef):
        """
        Add regularization gradient.
        """

        if self.penalty == "l2":

            grad_coef += 2 * self.alpha * self.coef_

        elif self.penalty == "l1":

            grad_coef += self.alpha * cp.sign(self.coef_)

        elif self.penalty == "elasticnet":

            grad_coef += self.alpha * (
                self.l1_ratio * cp.sign(self.coef_)
                + 2 * (1 - self.l1_ratio) * self.coef_
            )

        elif self.penalty is not None:

            raise ValueError("Invalid penalty.")

        return grad_coef

    def _get_learning_rate(self, t):
        """
        Compute learning rate for the current update.
        """

        if self.learning_rate == "constant":
            return self.eta0

        if self.learning_rate == "optimal":
            return 1.0 / (
                self.alpha * (t + self.t0)
            )

        if self.learning_rate == "invscaling":
            return self.eta0 / (t ** self.power_t)

        if self.learning_rate == "exponential":
            return self.eta0 * cp.exp(
                -self.decay * t
            )

        raise ValueError("Invalid learning rate.")

        # Validation
    
    def _validate_params(self):
        """
        Validate constructor parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if self.eta0 <= 0:
            raise ValueError("eta0 must be positive.")

        if self.power_t <= 0:
            raise ValueError("power_t must be positive.")

        if self.penalty not in {
            None,
            "l1",
            "l2",
            "elasticnet",
        }:
            raise ValueError("Invalid penalty.")

        if self.learning_rate not in {
            "constant",
            "optimal",
            "invscaling",
            "exponential",
        }:
            raise ValueError("Invalid learning_rate.")