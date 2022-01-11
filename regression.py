import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv('1.01. Simple linear regression.csv')
print(data.head(10))
print(data.info())
print(data.describe())

# Define the independent and the dependent variable
y = data['GPA']
x1 = data['SAT']

# Explore the data
plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize= 20)
plt.show()

# Regression
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())

# Explore the data
plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c = 'orange', label = 'regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize= 20)
plt.show()