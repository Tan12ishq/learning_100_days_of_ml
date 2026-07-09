
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from mlxtend.plotting import plot_decision_regions



# Load and preprocess dataset


df = pd.read_csv("../datasets/iris.csv")
df = df.drop("Id", axis=1)

x = df.drop("Species", axis=1)
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)



# Custom KNN Classifier


class CustomKNN:
    """
    Simple K-Nearest Neighbors classifier implemented from scratch.
    """

    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_samples = None
        self.n_features = None

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y).ravel()

        self.x_train = x
        self.y_train = y
        self.n_samples, self.n_features = x.shape

        if self.k > self.n_samples:
            raise ValueError(
                "k cannot be greater than the number of training samples."
            )

        return self

    def predict(self, x):
        x = np.asarray(x)
        predictions = []

        for sample in x:
            distances = np.sum((self.x_train - sample) ** 2, axis=1)

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]

            prediction = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, x, y):
        predictions = self.predict(x)
        return accuracy_score(y, predictions)



# Train Custom KNN


custom_knn = CustomKNN(k=5)
custom_knn.fit(x_train, y_train)

custom_predictions = custom_knn.predict(x_test)

print("Custom KNN Accuracy :", accuracy_score(y_test, custom_predictions))



# Compare with Scikit-learn KNN


sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(x_train, y_train)

sklearn_predictions = sklearn_knn.predict(x_test)

print("Scikit-learn Accuracy :", accuracy_score(y_test, sklearn_predictions))



# Decision Boundary (2 Features)


x_plot = df[["SepalLengthCm", "PetalLengthCm"]].values

encoder = LabelEncoder()
y_plot = encoder.fit_transform(df["Species"])

x_train_plot, x_test_plot, y_train_plot, y_test_plot = train_test_split(
    x_plot,
    y_plot,
    test_size=0.2,
    random_state=42
)

plot_knn = CustomKNN(k=5)
plot_knn.fit(x_train_plot, y_train_plot)

plt.figure(figsize=(8, 6))
plot_decision_regions(x_train_plot, y_train_plot, clf=plot_knn, legend=2)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Decision Boundary of Custom KNN")
plt.tight_layout()
plt.show()


# Confusion Matrix


ConfusionMatrixDisplay.from_predictions(
    y_test,
    custom_predictions,
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
