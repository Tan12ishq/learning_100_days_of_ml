import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
class multi_lr:
    def __init__(self):
        self.coeff=None
        self.intercept=None
    def fit(self,x,y):
        
        x=np.insert(x,0,1,axis=1)
        betas=np.linalg.inv(np.dot(x.T,x)).dot(x.T).dot(y)
        self.intercept=betas[0]
        self.coeff=betas[1:]
        
    def predict(self,x):
        y_pred=self.intercept+np.dot(x,self.coeff)
        return y_pred
    
x, y = make_regression(
    n_features=5,
    n_samples=10000,
    n_targets=1,
    n_informative=1,
    noise=20
)    

x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)

lr=multi_lr()
lr.fit(x_tr,y_tr)


y_pred=lr.predict(x_te)


model = LinearRegression()
model.fit(x_tr, y_tr)

print("My coefficients :", lr.coeff)
print("Sklearn coefficients:", model.coef_)

print("My intercept :", lr.intercept)
print("Sklearn intercept:", model.intercept_)
print("R2 score",r2_score(y_pred=y_pred,y_true=y_te))