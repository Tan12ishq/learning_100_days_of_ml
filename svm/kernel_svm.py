import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Generate Concentric Circle Dataset
# -----------------------------
x, y = make_circles(
    n_samples=1000,
    noise=0.2,
    factor=0.35,
    random_state=42
)

# -----------------------------
# Visualize Dataset
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Concentric Circle Dataset")
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
# Train SVM with RBF Kernel
# -----------------------------
svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

svm.fit(x_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = svm.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Create Mesh Grid
# -----------------------------
x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# -----------------------------
# Decision Regions
# -----------------------------
Z = svm.predict(grid)
Z = Z.reshape(xx.shape)

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
# Decision Boundary
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
plt.title("RBF Kernel SVM (Kernel Trick)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.show()