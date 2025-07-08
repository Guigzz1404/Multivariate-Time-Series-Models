import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.distributions import chi2


# Plot options
sns.set_theme(style="darkgrid")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))


# DATA TREATMENT
d_curr = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
d_curr = d_curr.loc[:, ["GBP", "NOK", "SEK", "CHF", "CAD", "SGD", "AUD"]]
d_curr = d_curr.dropna() # Data cleaning, remove all lines which contain NaN values
d_curr = d_curr.loc[d_curr.index <= "2011-10-26"] # Same range as GitHub
# Given most of the time series appear non-stationary, let's first attempt to difference them
d_curr_diff = d_curr.diff().dropna()

# Then, re-apply the ADF test to the differentiated series
adf_diff = []
for column in d_curr_diff:
    adf_test = adfuller(d_curr_diff[column], regression='ct')
    adf_diff.append(
        {
            "currency": column,
            "adf": adf_test[0],
            "p_value": adf_test[1]
        }
    )
adf_diff_df = pd.DataFrame(adf_diff)
print(adf_diff_df)

# Let's use the VAR model from statsmodels:
forecasting_model = VAR(d_curr_diff)

results_aic = []
for p in range(1,10):
    results = forecasting_model.fit(p)
    results_aic.append(results.aic)

# Let's plot the AIC curve to figure out the order _p_ of the model. The AIC is known to penalize model complexity, and
# its inflexion point represents the optimal order for the model
plt.plot(list(np.arange(1,10,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()

# The lowest AIC score is obtained for order 3, hence we choose VAR(3) as the optimal model
results = forecasting_model.fit(3)
print(results.summary())

# Let's compare log-likelihood for p=3 and p=2
resultsVAR1 = forecasting_model.fit(2)
print(results.llf)
print(resultsVAR1.llf)
# Model better with p=3

LR = likelihood_ratio(resultsVAR1.llf, results.llf)
p = chi2.sf(LR, 2) # L2 has 2 degree of freedom more than L1. sf: survival function to compare 2 models (H0 with fewer parameters than H1). If p is <0.05, we reject H0
print(f"p: {p:.30f}") # We print p is 30 decimals

# We reject HO because p is very small. So the model with 3 predictors is better than the one with 2.



