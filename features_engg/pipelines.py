import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder ,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
df=pd.read_csv("fifa_player_performance_market_value.csv")
df.drop(["player_id","player_name"],inplace=True,axis=1)
df.head()
df.info()
x_train,x_test,y_train,y_test=train_test_split(df.drop(['potential_rating'],axis=1),df["potential_rating"],test_size=0.1
                                               , random_state=0)
# conclusion- no null value, ohe - nationality ,club,postion 
#ordinal-injury,transfer
#encoding->scaling->feature extraction->train //pipeline

trf1=ColumnTransformer([
    ('ohe',OneHotEncoder(),[1,2,3]),
    ('ordi',OrdinalEncoder(categories=[['No','Yes'],['Low','Medium','High']]),[11,12])
    ],remainder="passthrough")

trf2=StandardScaler()

trf3= SelectKBest(score_func=f_regression,k=10)

trf4=DecisionTreeRegressor()
set_config(display='diagram')
pipe=Pipeline([
    ("trf1",trf1),('trf2',trf2),('trf3',trf3),('trf4',trf4)
])
pipe.fit(x_train,y_train)

y_pred=pipe.predict(x_test)

print(r2_score(y_test, y_pred))


