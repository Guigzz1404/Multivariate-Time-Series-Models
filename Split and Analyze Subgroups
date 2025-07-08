import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plot option
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def sub_group_stats(df):
    return pd.concat({"mean": df.mean(axis=0), "std": df.std(axis=0), "autocorr": df.apply(lambda s: s.autocorr(lag=1))}, axis=1)

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
all_ticker = list(stock_cols) + idx_cols

# Split df
# Let's split the time period in 4 sub-periods and create separate stats for each
date_split = np.array_split(daily_df.index.values, 4)
# Loop to browse each element in the list (i.e. arrays) and return i and array
for i, d_split in enumerate(date_split):
    daily_df.loc[d_split, "SubGroup"] = i+1
daily_df.SubGroup = daily_df.SubGroup.astype(int) # Convert to int the SubGroup column
daily_df = daily_df.set_index("SubGroup", append=True)
print(daily_df)

# List Comprehension to plot the subgroups date
sub_group_date = [(str(d.min())[:10], str(d.max())[:10]) for d in date_split]
print(sub_group_date)

# Stats calculation
sub_group_stats = daily_df.groupby("SubGroup").apply(sub_group_stats)
print(sub_group_stats)


# Plot statistics
# Plot means
fig1, ax1 = plt.subplots(figsize=(12,8))
sub_group_stats.loc[pd.IndexSlice[:, all_ticker], "mean"].unstack(level=1).plot(ax=ax1, marker="o")
_ = plt.xlabel("SubGroup")
_ = plt.ylabel("Means")
_ = plt.legend()
_ = plt.xticks(ticks = range(1,5), labels=[str(i) for i in range(1,5)]) # Change x graduation
_ = plt.xlim(0.6, 4.4) # Limits of x-axis

# Plot std
fig2, ax2 = plt.subplots(figsize=(12,8))
sub_group_stats.loc[pd.IndexSlice[:, all_ticker], "std"].unstack(level=1).plot(ax=ax2, marker="o")
_ = plt.xlabel("SubGroup")
_ = plt.ylabel("Standard Deviation")
_ = plt.legend()
_ = plt.xticks(ticks=range(1,5), labels=[str(i) for i in range(1,5)])
_ = plt.xlim(0.6, 4.4)

# Plot autocorr1
fig3, ax3 = plt.subplots(figsize=(12,8))
sub_group_stats.loc[pd.IndexSlice[:, all_ticker], "autocorr"].unstack(level=1).plot(ax=ax3, marker="o")
_ = plt.xlabel("SubGroup")
_ = plt.ylabel("Autocorrelation (1)")
_ = plt.legend()
_ = plt.xticks(ticks=range(1,5), labels=[str(i) for i in range(1,5)])
_ = plt.xlim(0.6, 4.4)

sns.despine()
plt.tight_layout()
plt.show()



