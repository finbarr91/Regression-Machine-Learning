import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

data = pd.read_csv('1.02. Multiple linear regression.csv')
print(data.head(10))
print(data.describe())

y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]


# Regression
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())