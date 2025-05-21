import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression

def linearRegression(x, y, sides=2):
    model1 = LinearRegression().fit(x, y)

    # Print model and p value
    slope = model1.coef_[0][0]
    print(f"Model: {slope}x + {model1.intercept_[0]}")
    residuals = y - model1.predict(x)
    S = np.std(residuals)
    df = len(x)-2
    Sx = np.std(x)
    SE = S/Sx/math.sqrt(df)
    t = slope/SE
    p = sides*(1 - stats.t.cdf(abs(t), df))
    print(f"p-value: {p}")

data = pd.read_csv('data.csv')

# Calculate win percentage and change in variables
data['Proportion'] = (data['Wins'] / (data['Wins']+data['Losses']+data["Ties"]))
data["RankChange"] = data["Rank2"] - data["Rank1"]
data["OPRChange"] = data["OPR2"] - data["OPR1"]

# Drop rows where OPRChange is 0
data2 = data.drop(data[data["OPRChange"] == 0].index)

x1 = data["Number"].values.reshape(-1, 1)
y1 = data["Proportion"].values.reshape(-1, 1)

print("\nTeam number vs win percentage")
linearRegression(x1, y1, 1)

x2 = data2["OPRChange"].values.reshape(-1, 1)
y2 = data2["RankChange"].values.reshape(-1, 1)
print("\nOPR change vs rank change")
linearRegression(x2, y2, 1)

y3 = data["EPA"].values.reshape(-1, 1)
print("\nTeam number vs EPA")
linearRegression(x1, y3, 1)

x3 = data["Points"].values.reshape(-1, 1)
print("\nPoints vs EPA")
linearRegression(x3, y3, 1)

x4 = data2["EPA"].values.reshape(-1, 1)
y4 = data2["OPRChange"].values.reshape(-1, 1)
y5 = data2["RankChange"].values.reshape(-1, 1)
print("\nEPA vs OPR change")
linearRegression(x4, y4, 1)
print("\nEPA vs rank change")
linearRegression(x4, y5, 1)

x5 = data2["Number"].values.reshape(-1, 1)
print("\nTeam number vs OPR change")
linearRegression(x5, y4, 1)

# Separate data into two groups based on district
district = data[data["District"] == 1]
regional = data[data["District"] == 0]
