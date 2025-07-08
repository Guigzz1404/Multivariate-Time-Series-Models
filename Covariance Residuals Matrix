import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


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

# VAR
forecasting_model = VAR(lret_curr)
resultsVAR5 = forecasting_model.fit(5)
# Residuals
curr_resid = resultsVAR5.resid

print(resultsVAR5.resid_acov(5))


