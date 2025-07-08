import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm


# Plot options
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 99% Confidence Interval function
def calc_confidence_interval(x):
    ci = DescrStatsW(x).tconfint_mean(alpha=0.01)
    mean = x.mean()
    return pd.Series({"lower_bound": ci[0], "mean": mean, "upper_bound": ci[1]})

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

# Split df
# Let's split the time period in 4 sub-periods and create separate stats for each
date_split = np.array_split(daily_df.index.values, 4)
# Loop to browse each element in the list (i.e. arrays) and return i and array
for i, d_split in enumerate(date_split):
    daily_df.loc[d_split, "SubGroup"] = i+1
daily_df.SubGroup = daily_df.SubGroup.astype(int) # Convert to int the SubGroup column
daily_df = daily_df.set_index("SubGroup", append=True)


# Entire Sample
# Confidence interval calculation
conf_interval = daily_df.apply(calc_confidence_interval).T
print(conf_interval)

# Plot of mean CI
fig, ax = plt.subplots(figsize=(12,8))
for i in range(conf_interval.shape[0]):
    colors = sns.color_palette(n_colors=conf_interval.shape[0])
    entry = conf_interval.iloc[i]
    plt.plot((i,i), (entry.lower_bound, entry.upper_bound), marker="_", color=cm.colors.to_rgba(colors[i]), linewidth=2.0, markersize=20.0, markeredgewidth=2.0)
    plt.plot(i, entry["mean"], marker=".", c=cm.colors.to_rgba(colors[i]), markersize=10.0)
plt.xticks(ticks=range(conf_interval.shape[0]), labels=conf_interval.index)


# SubGroups
# Confidence Interval Calculation
conf_interval_subgroup = daily_df.stack().groupby(level=[1,2]).apply(calc_confidence_interval).unstack(level=2)
print(conf_interval_subgroup)

# Plot of mean CI for subgroups
fig, ax = plt.subplots(figsize=(25,10))
for i in range(conf_interval_subgroup.index.get_level_values(1).nunique()):
    colors = sns.color_palette(n_colors=conf_interval_subgroup.index.get_level_values(0).nunique())
    for j in range(conf_interval_subgroup.index.get_level_values(0).nunique()):
        entry = conf_interval_subgroup.iloc[j*conf_interval_subgroup.index.get_level_values(1).nunique()+i]
        plt.plot((i,i), (entry.lower_bound, entry.upper_bound), marker="_", color=cm.colors.to_rgba(colors[j]), linewidth=2.0, markersize=20.0, markeredgewidth=2.0)
        plt.plot(i, entry["mean"], marker=".", c=cm.colors.to_rgba(colors[j]), markersize=10.0)
plt.xticks(ticks=range(conf_interval_subgroup.index.get_level_values(1).nunique()), labels=conf_interval_subgroup.index.get_level_values(1).unique())

sns.despine()
plt.tight_layout()
plt.show()



