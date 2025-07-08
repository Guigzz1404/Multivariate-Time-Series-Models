import pandas as pd
import numpy as np


# DATA EXPLORATION
# Overview of timeseries
d_15stock = pd.read_csv("d_15stocks.csv", index_col=["TICKER","date"], parse_dates=["date"])
# Extract RET, put index 0 (TICKER) as column (pivot operation), join d_index to the new df
daily_df = d_15stock.RET.unstack(level=0)
# Store stocks name and idx cols name
stock_cols = d_15stock.index.levels[0]

# We calculate the mean for each column, and we center each column around 0
C = daily_df - daily_df.mean()

# We calculate the covariance matrix of the returns
V = C.cov()

# We do the eigen decomposition
e_values, e_vectors = np.linalg.eig(V)

# Let's sort and normalizing eigenvalues
idx = np.argsort(e_values)[::-1] # Let's do argsort so we can then use it to choose the eigenvectors. argsort sorts ascending so we need to flip it with [::-1]
explained_variance = e_values[idx]/e_values.sum()

# Based on the graph, we should choose the first 3 factors
# Let's look at how much of the variance is explained
print(f"Explained Variance: {np.round(explained_variance[:3].sum()*100, 2)}%")

# We select the 3 eigenvectors associated with the 3 bigger eigenvalues, we transpose, and we apply a dot function with daily_df (matrix product)
pca_factors = daily_df.dot(e_vectors[idx[0:3]].T)
# The eigen vectors represent the "Factor Loadings" for each of the principal factors a portfolio with asset allocation

print(np.sqrt(np.cov(C.values[:,0],pca_factors[0])))