import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
df=pd.read_csv("../datasets/BudgetItaly.csv")
df.drop(df[['rownames']],inplace=True,axis=1)

#high corr with others
vif = pd.DataFrame()
x=df.drop(df[['totexp']],axis=1)

vif["Feature"] = x.columns
vif["VIF"] = [variance_inflation_factor(x.values, i)
              for i in range(x.shape[1])]

print(vif)
sns.heatmap(df.corr(), annot=True)
df.drop(df[['year','whouse','pmisc','pfood','size']],axis=1,inplace=True)
sns.heatmap(df.corr(), annot=True)
x=df.drop(df[['totexp']],axis=1)
y=df['totexp']

# linearity
features = x.columns

for col in features:
    plt.figure(figsize=(5,4))
    plt.scatter(df[col], y)
    plt.xlabel(col)
    plt.ylabel("Target")
    plt.title(f"{col} vs Target")
    plt.show()
# not good

#normal residue

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
scores = cross_val_score(lr, x_train, y_train, cv=20, scoring='r2')
print("Average R2:", scores.mean())

lr.coef_
lr.intercept_
residue = y_test - y_pred
sns.kdeplot(residue)        #comes out to be normal
sns.histplot(residue, kde=True)

# homoscdasticity

residue = y_test - y_pred

sns.scatterplot(x=y_pred, y=residue)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


#autocorr of error

plt.plot(residue)    #comes out to be random

#outliear detection

for col in x.columns:
    sns.boxplot(x=df[col])
    plt.show()          # lot of outliers in some columns
