from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score


# Load dataset
df = pd.read_csv("../datasets/iris.csv")

# Features and Target
x = df.drop("Species", axis=1)
y = df["Species"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Prediction
y_pred = gnb.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

score=cross_val_score(
    gnb,
    x,
    y,
    cv=10,
    scoring="accuracy"
)

print("Mean Accuracy:", score.mean())


