import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
df=pd.read_csv("used_car_price_prediction_1M.csv")
df=df.drop_duplicates()
# df.info()
(df.corr(numeric_only=True)>0.5)
df["Mileage_kmpl"] = df.groupby(["Brand","Model"])["Mileage_kmpl"].transform(lambda x: x.fillna(x.mean()))
df["Engine_CC"] = df.groupby(["Brand","Model"])["Engine_CC"].transform(lambda x: x.fillna(x.mean()))
df[["Engine_CC","Horsepower"]].corr()
engine=df["Engine_CC"]
horsepower=df["Horsepower"]
xtr = engine[horsepower.notna()]
ytr = horsepower[horsepower.notna()]
xte = engine[horsepower.isna()]
xtr=pd.DataFrame(xtr,columns=["Engine_CC"])
xte=pd.DataFrame(xte,columns=["Engine_CC"])
lr=LinearRegression()
lr.fit(xtr,ytr)
y_pred=lr.predict(xte)
df.loc[horsepower.isna(), "Horsepower"] = y_pred


mask = df["Color"].isna()
df.loc[mask, "Color"] = np.random.choice(
    df["Color"].dropna(),
    size=mask.sum(),
    replace=True
)

df["City"] = df["City"].fillna("Unknown")

mask = df["Transmission"].isna()
df.loc[mask, "Transmission"] = np.random.choice(
    df["Transmission"].dropna(),
    size=mask.sum(),
    replace=True
)
df["Fuel_Type"] = df["Fuel_Type"].replace(["electrik"], "Electric")


probs = df["Fuel_Type"].value_counts(normalize=True)

mask = df["Fuel_Type"].isna()

df.loc[mask, "Fuel_Type"] = np.random.choice(
    probs.index,
    size=mask.sum(),
    p=probs.values
)

df=df.drop(["Insurance_Valid","Service_History","Accidents","Tax_Paid","Number_of_Doors","Seats"],axis=1)

# transformation 

transformer=ColumnTransformer(transformers=[
    ("trf1",OneHotEncoder(drop="first"),["City","Color","Fuel_Type","Transmission"])
    ,('trf2',OrdinalEncoder(categories=[["First","Second","Third","Fourth+"]]),["Owner_Type"])
],remainder="passthrough")

encode_data=transformer.fit_transform(df)

feature_names = transformer.get_feature_names_out()

feature_names = [x.split("__")[-1] for x in feature_names]

encode_data = pd.DataFrame(
    transformer.fit_transform(df),
    columns=feature_names
)

# freqeuncy onehotencoding 
encode_data["Brand_Model"] = encode_data["Brand"] + "_" + encode_data["Model"]
threshold = 1000
freq_brand_model= encode_data["Brand_Model"].value_counts()

encode_data["Brand_Model"] = encode_data["Brand_Model"].apply(
    lambda x: x if freq_brand_model[x] >= threshold else "Other"
)

transformer=ColumnTransformer(transformers=[
    ("trf1",OneHotEncoder(drop="first"),["Brand_Model"])
],remainder="passthrough")

freq_encode_data=transformer.fit_transform(encode_data)

feature_names = transformer.get_feature_names_out()


feature_names = [x.split("__")[-1] for x in feature_names]
freq_encode_data = pd.DataFrame(
    transformer.fit_transform(encode_data),
    columns=feature_names
)
freq_encode_data=freq_encode_data.drop(["Brand","Model"],axis=1)

# just checking data
corr = freq_encode_data.corr()['Price']

corr[corr.abs() > 0.2].sort_values(key=abs, ascending=False)


freq_encode_data.drop("Year", axis=1, inplace=True) #corr with reg_age was -1 so avoid multicollinearity

from sklearn.ensemble import HistGradientBoostingRegressor 

model = HistGradientBoostingRegressor(random_state=42)
freq_encode_data["Price"] = pd.to_numeric(
    freq_encode_data["Price"],
    errors="coerce"
)
x = freq_encode_data.drop("Price", axis=1)

y = np.log1p(freq_encode_data["Price"])
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R²:", r2_score(y_test, pred))


from sklearn.inspection import permutation_importance

X_small = X_test.sample(3000, random_state=42)
y_small = y_test.loc[X_small.index]

result = permutation_importance(
    model,
    X_small,
    y_small,
    n_repeats=2,
    random_state=42,
    n_jobs=-1
)



importance = pd.Series(
    result.importances_mean,
    index=x.columns
).sort_values(ascending=False)

print(importance.head(20))
print("R²:", r2_score(y_test, pred))


