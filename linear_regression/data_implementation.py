import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# wfood=percentage of total expenditure which the household has spent on food
# totexp=total expenditure of the household
# town=size of the town where the household is placed categorized into 5 groups: 1 for small towns, 5 for big ones

df = pd.read_csv("../datasets/BudgetFood.csv")
df.drop(df[["rownames", "town", "sex"]], axis=1, inplace=True)
df.corr()
x_train, x_test, y_train, y_test = train_test_split(
    df.drop("wfood", axis=1), df["wfood"], test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)



r2 = r2_score(y_test, y_pred)

n = len(y_test)          
p = x_test.shape[1]      

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("R²:", r2)
print("Adjusted R²:", adj_r2)