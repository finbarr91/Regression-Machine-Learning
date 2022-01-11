import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.01. Simple linear regression.csv')
print(data.head())

x = data['SAT']
y = data['GPA']

print(x.shape)
print(y.shape)

# Reshaping the dimension of the feature so as to fit in the model (sklearn uses matrix and not arrays)
x_matrix = x.values.reshape(-1,1)
# model
reg = LinearRegression()
reg.fit(x_matrix,y)

# R-squared
print(reg.score(x_matrix,y))

# Coefficients
print(reg.coef_)

# Intercept
print(reg.intercept_)

# Making predictions
new_data = pd.DataFrame(data=[1740,1760], columns=['SAT'])
new_data['Predicted_GPA'] = reg.predict(new_data)

print(reg.predict(new_data))
