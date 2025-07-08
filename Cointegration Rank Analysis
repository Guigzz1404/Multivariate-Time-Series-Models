import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_coint_rank


# DATA TREATMENT
d_curr = pd.read_csv("exchange_rates_2.csv", index_col="DATE", parse_dates=["DATE"])
d_curr = d_curr.loc[:, ["GBP", "NOK", "SEK", "CHF", "CAD", "SGD", "AUD"]]
d_curr = d_curr.dropna() # Data cleaning, remove all lines which contain NaN values
d_curr = d_curr.loc[d_curr.index <= "2011-10-26"] # Same range as GitHub
# Given most of the time series appear non-stationary, let's first attempt to difference them
d_curr_diff = d_curr.diff().dropna()

# Let's analyze the co-integration rank of the differenced time series
coint_rank = select_coint_rank(d_curr_diff, det_order=-1, k_ar_diff=1) # det_order=-1 implies auto constant and trend
print(f"Johansen cointegration rank: {coint_rank.rank}")
print(f"Johansen cointegration test results: ")
print(coint_rank.summary())

# Let's analyze the co-integration rank of the exchange rate time series
coint_rank = select_coint_rank(d_curr, det_order=-1, k_ar_diff=1)
print(f"Johansen cointegration rank: {coint_rank.rank}")
print(f"Johansen cointegration test results: ")
print(coint_rank.summary())
# Unsurprisingly, the non-differenced series are not co-integrated (bc no stationary)




