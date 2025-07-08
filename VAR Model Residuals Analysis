import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from statsmodels.stats.stattools import jarque_bera


# Display option
pd.options.display.max_columns = None
sns.set_theme(style="darkgrid")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


# DATA TREATMENT
d_curr = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
d_curr = d_curr.loc[:, ["GBP", "NOK", "SEK", "CHF", "CAD", "SGD", "AUD"]]
d_curr = d_curr.dropna() # Data cleaning, remove all lines which contain NaN values
d_curr = d_curr.loc[d_curr.index <= "2011-10-26"] # Same range as GitHub

# Log return differenced series
for column in d_curr:
    colname = str(column + "lret")
    d_curr[colname] = np.log(d_curr[column]) - np.log(d_curr[column].shift(1))
lret_curr = d_curr.loc[:, ["GBPlret", "NOKlret", "SEKlret", "CHFlret", "CADlret", "SGDlret", "AUDlret"]].dropna()
print(lret_curr.head())

# Let's do a quick sanity check with the ADF test
adf_lret = []
for column in lret_curr:
    adf_test = adfuller(lret_curr[column], regression="ct")
    adf_lret.append(
        {
            "currency": column,
            "adf": adf_test[0],
            "p-value": adf_test[1]
        }
    )
adf_lret_df = pd.DataFrame(adf_lret)
print(adf_lret_df)

# The series appear to be stationary, so we can fit a VAR model
# First, we check the optimal order
forecasting_model = VAR(lret_curr)
results_aic = []
for p in range (1,10):
    results = forecasting_model.fit(p)
    results_aic.append(results.aic)
plt.plot(np.arange(1,10), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC") # Optimal order with AIC: 3

# The exercise asks for a VAR(5)
resultsVAR5 = forecasting_model.fit(5)
# resultsVAR3 = forecasting_model.fit(3)
# print(resultsVAR5.llf)
# print(resultsVAR3.llf) # Optimal order with llf: 5
print(resultsVAR5.summary())

# Let's extract the residuals:
curr_resid = resultsVAR5.resid
print(curr_resid.head())

# Let's plot residual to see if it's a white noise
fig, ax = plt.subplots(figsize=(12,8))
_ = curr_resid.plot(ax=ax)
_ = plt.xlabel('Date')
_ = plt.ylabel('Residuals')
_ = plt.legend()
sns.despine()
plt.tight_layout()
plt.show() # It seems to be a white noise, but let's test it with Jarque-Bera test

# Let's test this assumption
jarquebera_results = curr_resid[dt.datetime(2002, 1, 1, 0, 0) : dt.datetime(2006, 12, 31, 0, 0)].apply(lambda x: jarque_bera(x), axis=0)
jarquebera_results = jarquebera_results.T.rename(columns={0: "JB_test", 1: "p_value", 2: "skew", 3: "kurtosis"})
print(jarquebera_results)
# The Jarque-Bera test at 95% rejects the normality of the individual time series of residuals, even on the more consistent regime of 2002-2006.


