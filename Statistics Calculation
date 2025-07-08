import pandas as pd

# Data Treatment
d_15stock = pd.read_csv("d_15stocks.csv", index_col=["TICKER","date"], parse_dates=["date"])
m_15stock = pd.read_csv("m_15stocks.csv", index_col=["TICKER", "date"], parse_dates=["date"])
d_indexes = pd.read_csv("d_indexes.csv", index_col=["caldt"], parse_dates=["caldt"])
d_indexes.index.name = "date"
m_indexes = pd.read_csv("m_indexes.csv", index_col=["caldt"], parse_dates=["caldt"])
m_indexes.index.name = "date"
# Put columns in uppercase
d_indexes.columns = d_indexes.columns.str.upper()
m_indexes.columns = m_indexes.columns.str.upper()

# Extract RET, put index 0 (TICKER) as column (pivot operation), join d_index to the new df
daily_df = d_15stock.RET.unstack(level=0).join(d_indexes)
# print(daily_df.head())
# Store stocks name and idx cols name
stock_cols = d_15stock.index.levels[0]
idx_cols = d_indexes.columns.tolist()

# Let's review the main statistics of the returns:
stats = pd.concat({"mean": daily_df.mean(axis=0), "std": daily_df.std(axis=0), "autocorr": daily_df.apply(lambda s: s.autocorr(lag=1))}, axis=1)
print(stats)



