import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


# Display settings
pd.options.display.max_columns = None
pd.set_option('display.max_rows', None)
pd.options.display.width=None
sns.set_theme(style="darkgrid")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# DATA TREATMENT
d_curr = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
d_curr = d_curr.loc[:, ["GBP", "NOK", "SEK", "CHF", "CAD", "SGD", "AUD"]]
d_curr = d_curr.dropna() # Data cleaning, remove all lines which contain NaN values
d_curr = d_curr.loc[d_curr.index <= "2011-10-26"] # Same range as GitHub
print(d_curr.info())

# Before running the ADF test, we need to inspect the time series to get a sense of the appropriate regression model
# Data plot
fig1, ax1 = plt.subplots(figsize=(12,8))
_ = d_curr.loc[:, ["GBP", "CHF", "CAD", "SGD", "AUD"]].plot(ax=ax1)
_ = plt.ylabel("Fx Rates")
_ = plt.xlabel("Date")
_ = plt.title("Fx Rates of GBP, CHF, CAD, SGD and AUD", fontsize=16)
sns.despine()
plt.tight_layout()
fig2, ax2 = plt.subplots(figsize=(12,8))
_ = d_curr.loc[:, ["NOK", "SEK"]].plot(ax=ax2)
_ = plt.ylabel("Fx Rates")
_ = plt.xlabel("Date")
_ = plt.title("Fx Rates of NOK and SEK", fontsize=16)
sns.despine()
plt.tight_layout()
plt.show()

# Run ADF test for each of the currencies

# Note, the statsmodels package returns the following:
# The ADF value is the first value in the result and the p-value is the 2nd.
# The ‘1%’, ‘10%’ and ‘5%’ values are the critical values at 99% (-3.96), 95% (-3.41) and 90% (-3.12) confidence levels for large sample size

adf = []
for column in d_curr:
    adf_test = adfuller(d_curr[column], regression="ct")
    adf.append(
        {
            "currency": column,
            "adf": adf_test[0],
            "p-value": adf_test[1]
        }
    )

adf_df = pd.DataFrame(adf)
print(adf_df)

