import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB

import matplotlib.pyplot as plt


# Load Dataset

df = pd.read_csv("../datasets/PlayTennis.csv")

# Features and Target

x = df.drop("play", axis=1)
y = df["play"]

# Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0,
)


class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Categorical Naive Bayes classifier implemented from scratch
    using Laplace smoothing.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_count = {}
        self.class_prior = {}
        self.feature_probs = {}
        self.predictions = None

    def fit(self, x, y):
        """Train the Naive Bayes classifier."""

        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        self.class_count = {}
        self.class_prior = {}
        self.feature_probs = {}

        # Compute class prior probabilities
        for cls in self.classes_:
            self.class_count[cls] = np.sum(y == cls)
            self.class_prior[cls] = np.log(self.class_count[cls] / len(y))

        # Compute feature likelihoods
        for feature in x.columns:

            self.feature_probs[feature] = {}

            num_values = x[feature].nunique()

            for cls in self.classes_:

                self.feature_probs[feature][cls] = {}

                class_data = x[y == cls]
                value_counts = class_data[feature].value_counts()

                total = len(class_data)

                for value in x[feature].unique():

                    count = value_counts.get(value, 0)

                    self.feature_probs[feature][cls][value] = np.log(
                        (count + self.alpha)
                        / (total + self.alpha * num_values)
                    )

        return self

    def predict(self, x):
        """Predict class labels."""

        self.predictions = []

        for _, row in x.iterrows():

            class_scores = {}

            for cls in self.classes_:

                log_probability = self.class_prior[cls]

                for feature in x.columns:

                    value = row[feature]

                    log_probability += self.feature_probs[feature][cls].get(
                        value,
                        np.log(
                            self.alpha
                            / (
                                self.class_count[cls]
                                + self.alpha * len(self.feature_probs[feature][cls])
                            )
                        ),
                    )

                class_scores[cls] = log_probability

            self.predictions.append(max(class_scores, key=class_scores.get))

        return np.array(self.predictions)

    def predict_proba(self, x):
        """Return class probabilities."""

        all_probabilities = []

        for _, row in x.iterrows():

            class_scores = []

            for cls in self.classes_:

                log_probability = self.class_prior[cls]

                for feature in x.columns:

                    value = row[feature]

                    log_probability += self.feature_probs[feature][cls].get(
                        value,
                        np.log(
                            self.alpha
                            / (
                                self.class_count[cls]
                                + self.alpha * len(self.feature_probs[feature][cls])
                            )
                        ),
                    )

                class_scores.append(log_probability)

            class_scores = np.array(class_scores)
            class_scores = np.exp(class_scores - np.max(class_scores))
            class_scores /= np.sum(class_scores)

            all_probabilities.append(class_scores)

        return np.array(all_probabilities)

    def predict_log_proba(self, x):
        """Return log probabilities."""
        return np.log(self.predict_proba(x))

    def score(self, x, y):
        """Return classification accuracy."""
        return accuracy_score(y, self.predict(x))


# My Model

my_nb = CategoricalNaiveBayes(alpha=1.0)
my_nb.fit(x_train, y_train)

my_pred = my_nb.predict(x_test)

print("My Model Accuracy:", accuracy_score(y_test, my_pred))

# Sklearn Model

x_train_encoded = x_train.copy()
x_test_encoded = x_test.copy()

for column in x.columns:

    categories = pd.Categorical(
        pd.concat([x_train[column], x_test[column]])
    ).categories

    x_train_encoded[column] = pd.Categorical(
        x_train[column],
        categories=categories,
    ).codes

    x_test_encoded[column] = pd.Categorical(
        x_test[column],
        categories=categories,
    ).codes

sklearn_nb = CategoricalNB(alpha=1.0)
sklearn_nb.fit(x_train_encoded, y_train)

sklearn_pred = sklearn_nb.predict(x_test_encoded)

print("Sklearn Accuracy:", accuracy_score(y_test, sklearn_pred))

# Cross Validation

cv_scores = cross_val_score(
    CategoricalNaiveBayes(alpha=1.0),
    x,
    y,
    cv=3,
    scoring="accuracy",
)

print("Mean CV Accuracy:", cv_scores.mean())

# Confusion Matrix

ConfusionMatrixDisplay.from_predictions(y_test, my_pred)

plt.title("Confusion Matrix - My Naive Bayes")
plt.show()
