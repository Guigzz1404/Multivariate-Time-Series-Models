import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plot options
sns.set_theme(style="darkgrid")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Data treatment
d_indexes = pd.read_csv("d_indexes.csv", index_col=["caldt"], parse_dates=["caldt"])
d_indexes.index.name = "date"
# Put columns in uppercase
d_indexes.columns = d_indexes.columns.str.upper()

# Normal distribution generation
means = d_indexes.mean(axis=0)
stds = d_indexes.std(axis=0)
n_points = d_indexes.shape[0]
idx_1 = pd.Series(np.random.normal(means.iloc[0], stds.iloc[0], n_points), name="Normal")
idx_2 = pd.Series(np.random.normal(means.iloc[1], stds.iloc[1], n_points), name="Normal")

# Let's inspect the distributions and compare to a Normal
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8), sharey=True)
_ = sns.histplot(d_indexes.VWRETD, ax=ax1, stat="density", bins=50, kde=True)
_ = sns.histplot(idx_1, ax=ax2, stat="density", bins=50, kde=True)
_ = plt.xlim(-0.12, 0.12)
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12,8), sharey=True)
_ = sns.histplot(d_indexes.EWRETD, ax=ax3, stat="density", bins=50, kde=True)
_ = sns.histplot(idx_2, ax=ax4, stat="density", bins=50, kde=True)
_ = plt.xlim(-0.12, 0.12)

sns.despine()
plt.tight_layout()
plt.show()