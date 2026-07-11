import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Generate Dataset
# -----------------------------
x, y = make_blobs(
    n_samples=1000,
    centers=2,
    cluster_std=2,
    random_state=42
)

# -----------------------------
# Visualize Dataset
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Linearly Separable Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# -----------------------------
# Train-Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Linear SVM
# -----------------------------
svm = SVC(kernel="linear")
svm.fit(x_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = svm.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Create Mesh Grid
# -----------------------------
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# -----------------------------
# Predict Every Grid Point
# -----------------------------
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm.predict(grid)
Z = Z.reshape(xx.shape)
# -----------------------------
# Plot Decision Regions
# -----------------------------
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    cmap="bwr"
)

# -----------------------------
# Plot Training Points
# -----------------------------
plt.scatter(
    x_train[:, 0],
    x_train[:, 1],
    c=y_train,
    cmap="bwr",
    edgecolors="k",
    s=40
)

# -----------------------------
# Highlight Support Vectors
# -----------------------------
plt.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidth=2,
    label="Support Vectors"
)

# -----------------------------
# Decision Boundary & Margins
# -----------------------------
Z = svm.decision_function(grid)
Z = Z.reshape(xx.shape)

plt.contour(
    xx,
    yy,
    Z,
    levels=[-1, 0, 1],
    colors="black",
    linestyles=["--", "-", "--"],
    linewidths=2
)

# -----------------------------
# Labels
# -----------------------------
plt.title("Linear SVM Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.show()