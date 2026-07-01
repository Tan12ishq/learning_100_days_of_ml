import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../datasets/BudgetItaly.csv")
df
x = df["whouse"]
y = df["totexp"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = pd.DataFrame(x_train, columns=["whouse"])
y_train = pd.DataFrame(y_train, columns=["totexp"])
x_test = pd.DataFrame(x_train, columns=["whouse"])
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
lr.intercept_
lr.coef_


class gd:
    def __init__(self):
        self.b = 0
        self.m = 1
        self.lr = 0.01

    def cal(self, x, y):
        x = x.to_numpy()
        y = y.to_numpy()
        for i in range(10000):
            y_pred = self.m * x + self.b

            dm = (-2 / len(x)) * np.sum(x * (y - y_pred))
            db = (-2 / len(x)) * np.sum(y - y_pred)

            self.m -= self.lr * dm
            self.b -= self.lr * db
        return self.b,self.m


g = gd()
b,m=g.cal(x_train, y_train)

print("My intercept:",b)
print("skicit intercep:",lr.intercept_)
print("My slope:",m)
print("skicit slope:",lr.coef_)