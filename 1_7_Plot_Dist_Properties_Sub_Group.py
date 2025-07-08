import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plot options
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def calc_distributional_properties(x):
    kurtosis = x.kurtosis()
    skew = x.skew()
    sr = (x.max() - x.min())/x.std()
    cnt = len(x)
    return pd.Series({"skew": skew, "kurtosis": kurtosis, "studentized range": sr, "count": cnt})
# Note: be careful, the 'kurtosis' method returns the Fisher kurtosis, which is the Pearson kurtosis subtracted by 3

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
# Store stocks name and idx cols name
stock_cols = d_15stock.index.levels[0]
idx_cols = d_indexes.columns.tolist()

# Split df
# Let's split the time period in 4 sub-periods and create separate stats for each
date_split = np.array_split(daily_df.index.values, 4)
# Loop to browse each element in the list (i.e. arrays) and return i and array
for i, d_split in enumerate(date_split):
    daily_df.loc[d_split, "SubGroup"] = i+1
daily_df.SubGroup = daily_df.SubGroup.astype(int) # Convert to int the SubGroup column
daily_df = daily_df.set_index("SubGroup", append=True)

# Normal distribution generation
means = d_indexes.mean(axis=0)
stds = d_indexes.std(axis=0)
n_points = d_indexes.shape[0]
idx_2 = pd.Series(np.random.normal(means.iloc[1], stds.iloc[1], n_points), name="Normal")

# Let's look at log returns too as a comparison (more normal than ["RET"] but still have excess kurtosis)
daily_df_log = daily_df.apply(lambda x: np.log1p(x))
# Plot DD distribution for log return
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8), sharey=True)
_ = sns.histplot(daily_df_log["DD"], ax=ax1, stat="density", bins=50, kde=True)
_ = sns.histplot(idx_2, ax=ax2, stat="density", bins=50, kde=True)
_ = plt.xlim(-0.12, 0.12)

# Dist properties calculation for subgroups
subgrp_stats_df = daily_df.stack().groupby(level=[1,2]).apply(calc_distributional_properties).unstack(level=2)
# Plot dist properties
fig2, ax3 = plt.subplots(3, 1, figsize=(12,8*3))
for i, stat in enumerate(["skew", "kurtosis", "studentized range"]):
    subgrp_stats_df.loc[pd.IndexSlice[:, stock_cols],stat].unstack(level=1).plot(ax = ax3[i],marker='o')
    _ = ax3[i].set_ylabel(stat.capitalize())
    _ = ax3[i].legend()
    _ = ax3[i].axes.set_xticklabels(labels=[str(i) for i in range(1, 5)])
    _ = ax3[i].set_xticks(range(1, 5))
    _ = ax3[i].set_xlim(0.6, 4.4)
sns.despine()
plt.tight_layout()

fig3,ax4 = plt.subplots(3,1,figsize=(12,8*3))
for i, stat in enumerate(['skew','kurtosis','studentized range']):
    subgrp_stats_df.loc[pd.IndexSlice[:,idx_cols],stat].unstack(level=1).plot(ax = ax4[i],marker='o')
    _ = ax4[i].set_xlabel('SubGroup')
    _ = ax4[i].set_ylabel(stat.capitalize())
    _ = ax4[i].legend()
    _ = ax4[i].axes.set_xticklabels(labels = [str(i) for i in range(1,5) ])
    _ = ax4[i].set_xticks(range(1,5))
    _ = ax4[i].set_xlim(0.6,4.4)

sns.despine()
plt.tight_layout()
plt.show()

daily_s_df = daily_df.loc[pd.IndexSlice[:,stock_cols]].reset_index(level=0,drop=True)
C = daily_s_df - daily_s_df.mean()
V = C.cov()
print(V)







