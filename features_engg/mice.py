import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df = sns.load_dataset("titanic")
x=df[["age",'fare','pclass']]
y=df['survived']
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.1,random_state=42)
# while mice
mice=IterativeImputer(
    random_state=42,
    max_iter=20,
    initial_strategy="mean",
    add_indicator=True
)
col = np.append(xtr.columns, "missing")
xtr_m=mice.fit_transform(xtr)
xte_m=mice.transform(xte)
xtr_m=pd.DataFrame(xtr_m,columns=col)
xte_m=pd.DataFrame(xte_m,columns=col)

model = LogisticRegression(max_iter=1000)
model.fit(xtr_m, ytr)
y_pred = model.predict(xte_m)

print("Accuracy:", accuracy_score(yte, y_pred))
print(classification_report(yte, y_pred))

# without mice

si=SimpleImputer(add_indicator=True)
xtr_s=si.fit_transform(xtr)
xte_s=si.transform(xte)
xtr_s=pd.DataFrame(xtr_s,columns=col)
xte_s=pd.DataFrame(xte_s,columns=col)

model2 = LogisticRegression(max_iter=1000)
model2.fit(xtr_s, ytr)
y_pred_s = model2.predict(xte_s)
print("Accuracy:", accuracy_score(yte, y_pred_s))
print(classification_report(yte, y_pred_s))