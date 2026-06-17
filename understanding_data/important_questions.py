import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("crypto_daily.csv")
df
df['date'] = pd.to_datetime(df['date'])
df['date'] 
bit=df[df["symbol"]=="BTC"]
bit.shape
bit.info()
bit.describe()

sns.scatterplot(data=bit, x="date", y="open")
plt.show()
sns.displot(bit["open"],kde=True)
sns.distplot(bit[bit["is_weekend"]==1]["open"],hist=False)
sns.distplot(bit[bit["is_weekend"]==0]["open"],hist=False,color="red")

bit["open"].skew()
bit
bit.sample(5)
sns.scatterplot(bit,y="btc_miner_revenue_usd",x="year")
sns.barplot(bit,y="btc_miner_revenue_usd",x="year")
sns.barplot(bit,x="date",y="open")


sns.pairplot(bit)
sns.lineplot(bit,x="date",y='open')