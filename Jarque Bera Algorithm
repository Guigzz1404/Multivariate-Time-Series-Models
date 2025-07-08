import pandas as pd


def calc_distributional_properties(x):
    kurtosis = x.kurtosis()
    skew = x.skew()
    sr = (x.max() - x.min())/x.std()
    cnt = len(x)
    return pd.Series({"skew": skew, "kurtosis": kurtosis, "studentized range": sr, "count": cnt})
# Note: be careful, the 'kurtosis' method returns the Fisher kurtosis, which is the Pearson kurtosis subtracted by 3

def JB(df):
    return (df["count"]/6) * (df["skew"]**2 + 0.25*(df["kurtosis"])**2)


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

# Extract RET, put index 0 (TICKER) as column (pivot operation), join d_index to the new df
daily_df = d_15stock.RET.unstack(level=0).join(d_indexes)

# Dist properties calc
stats_df = daily_df.apply(calc_distributional_properties).T
# Jarque Bera test
stats_df["JB_test"] = stats_df.apply(lambda x: JB(x), axis=1)
print(stats_df)







