import pandas as pd

df=pd.read_csv("used_car_price_prediction_1M.csv")
df = df.dropna(subset=['Mileage_kmpl', 'Engine_CC'])
data=df[['Mileage_kmpl', 'Engine_CC',"Price"]]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data[['Mileage_kmpl', 'Engine_CC']],
                                               data.drop(['Mileage_kmpl', 'Engine_CC'],axis=1),
                                               test_size=0.1
                                               , random_state=0
                                               )

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)

xs_train=scaler.transform(x_train)
xs_test=scaler.transform(x_test)


xs_train=pd.DataFrame(xs_train,columns=x_train.columns)
xs_test=pd.DataFrame(xs_test,columns=x_test.columns)



from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lrs = LinearRegression()
lr.fit(x_train, y_train)
lrs.fit(xs_train, y_train)

y_pred=lr.predict(x_test)
ys_pred=lrs.predict(xs_test)

from sklearn.metrics import r2_score, mean_absolute_error
print("Without scaling")
print("R2 :", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print(y_train.describe())


print("With scaling")
print("R2 :", r2_score(y_test, ys_pred))
print("MAE:", mean_absolute_error(y_test, ys_pred))


print(data.corr(numeric_only=True))
