import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
df = pd.read_csv("crypto_daily.csv")
df["date"] = pd.to_datetime(df["date"])

# BTC only
bit = df[df["symbol"] == "BTC"].copy()

# Sort by date (VERY IMPORTANT)
bit = bit.sort_values("date")

# Create tomorrow's high
bit["next_high"] = bit["high"].shift(-1)

# Remove last row (next_high will be NaN there)
bit = bit.dropna()

# Features (today's values)
X = bit[["open"]]

# Target (tomorrow's high)
y = bit["next_high"]

# Time-series split
split = int(len(bit) * 0.8)

x_train = X.iloc[:split]
x_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Scaling
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Models
lr = LinearRegression()
lr_scaled = LinearRegression()

# Train
lr.fit(x_train, y_train)
lr_scaled.fit(x_train_scaled, y_train)

# Predict
y_pred = lr.predict(x_test)
y_pred_scaled = lr_scaled.predict(x_test_scaled)

# Scores
print("Without scaling")
print("R2 :", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

print()

print("With scaling")
print("R2 :", r2_score(y_test, y_pred_scaled))
print("MAE:", mean_absolute_error(y_test, y_pred_scaled))

# Correlation matrix
print("\nCorrelation Matrix:")
print(bit[["open", "close", "adj_close", "high", "next_high"]].corr())