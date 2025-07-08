import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Plot option
pd.options.display.max_columns = None
pd.options.display.width=None
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# DATA EXPLORATION
# Overview of timeseries
d_15stock = pd.read_csv("d_15stocks.csv", index_col=["TICKER","date"], parse_dates=["date"])
m_15stock = pd.read_csv("m_15stocks.csv", index_col=["TICKER", "date"], parse_dates=["date"])
d_indexes = pd.read_csv("d_indexes.csv", index_col=["caldt"], parse_dates=["caldt"])
d_indexes.index.name = "date"
m_indexes = pd.read_csv("m_indexes.csv", index_col=["caldt"], parse_dates=["caldt"])
m_indexes.index.name = "date"
# Put columns in uppercase
d_indexes.columns = d_indexes.columns.str.upper()
m_indexes.columns = m_indexes.columns.str.upper()

# print(d_15stock)
# print(m_15stock)
# print(d_indexes)
# print(m_indexes)

# Extract RET, put index 0 (TICKER) as column (pivot operation), join d_index to the new df
daily_df = d_15stock.RET.unstack(level=0).join(d_indexes)
print(daily_df.head())
# Store stocks name and idx cols name
stock_cols = d_15stock.index.levels[0]
idx_cols = d_indexes.columns.tolist()

# Let's inspect some time series as an example:
fig1,ax1 = plt.subplots(figsize=(12,8))
_ = daily_df.MSFT.plot(ax=ax1)
_ = daily_df.VWRETD.plot(ax=ax1)
_ = plt.xlabel("Date")
_ = plt.ylabel("Returns")
_ = plt.legend()

# Let's inspect distributions for outliers
fig2,ax2 = plt.subplots(2, 2, figsize=(12,8))
fig3 = sns.histplot(daily_df["JPM"], bins=100, ax=ax2[0][0], stat="density", kde=True)
fig4 = sns.histplot(daily_df["MSFT"], bins=100, ax=ax2[0][1], stat="density", kde=True)
fig5 = sns.histplot(daily_df["VWRETD"], bins=100, ax=ax2[1][0], stat="density", kde=True)
fig6 = sns.histplot(daily_df["EWRETD"], bins=100, ax=ax2[1][1], stat="density", kde=True)

# Let's also inspect joint distributions:
# Here kde means kernel density plots. We compare MSFT with JPM
# How their distribution of return evolve ? If circle it's independant, if diagonal it's evolve together
sns.jointplot(data=daily_df, x="MSFT", y="JPM", kind='kde').fig.set_size_inches(12, 8)

# Finally, let's inspect the scatter matrix plot for some of the tickers to understand the data structure better
# We inspect the joint distribution for several asset.
sns.pairplot(daily_df.loc[:,['T','VZ','WMT','VWRETD']]).fig.set_size_inches(12, 8)

sns.despine()
plt.tight_layout()
plt.show()



