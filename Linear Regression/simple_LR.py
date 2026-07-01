from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
#creating simple LR class
class Simple_lr:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, x, y):
        num = 0
        den = 0

        x_mean = x.mean()
        y_mean = y.mean()

        for i in range(x.shape[0]):
            num += (x[i] - x_mean) * (y[i] - y_mean)
            den += (x[i] - x_mean) ** 2

        self.m = num / den
        self.b = y_mean - self.m * x_mean

    def predict(self, x):
        return self.m * x + self.b





x, y = make_regression(
    n_features=1,
    n_samples=10000,
    n_targets=1,
    n_informative=1,
    noise=20
)    
x = x.flatten()
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)

lr=Simple_lr()
lr.fit(x_tr,y_tr)


y_pred=lr.predict(x_te)




model = LinearRegression()
model.fit(x_tr.reshape(-1, 1), y_tr)

print("My slope :", lr.m)
print("Sklearn slope:", model.coef_[0])

print("My intercept :", lr.b)
print("Sklearn intercept:", model.intercept_)
print("R2 score",r2_score(y_pred=y_pred,y_true=y_te))