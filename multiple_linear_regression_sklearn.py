# Multiple linear regression
# Import the relevant libraries

# For these lessons we will need NumPy, pandas, matplotlib and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.feature_selection import f_regression

# and of course the actual regression (machine learning) module
from sklearn.linear_model import LinearRegression

# Load the data from a .csv in the same folder
data = pd.read_csv('1.02. Multiple linear regression.csv')

# Let's explore the top 5 rows of the df
print(data.head())

# This method gives us very nice descriptive statistics. We don't need this for now, but will later on!
print(data.describe())

# Create the multiple linear regression
# Declare the dependent and independent variables

# There are two independent variables: 'SAT' and 'Rand 1,2,3'
x = data[['SAT','Rand 1,2,3']]

# and a single depended variable: 'GPA'
y = data['GPA']

# Regression itself

# We start by creating a linear regression object
reg = LinearRegression()

# The whole learning process boils down to fitting the regression
reg.fit(x,y)

# Getting the coefficients of the regression
print(reg.coef_)
# Note that the output is an array

# Getting the intercept of the regression
print(reg.intercept_)
# Note that the result is a float as we usually expect a single value

# Calculating the R-squared
# Get the R-squared of the regression
print(reg.score(x,y))

# Get the shape of x, to facilitate the creation of the Adjusted R^2 metric
print(x.shape)

# If we want to find the Adjusted R-squared we can do so by knowing the r2, the # observations, the # features
r2 = reg.score(x,y)
# Number of observations is the shape along axis 0
n = x.shape[0]
# Number of features (predictors, p) is the shape along axis 1
p = x.shape[1]

# We find the Adjusted R-squared using the formula
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print(adjusted_r2)

# Feature selection

# Import the feature selection module from sklearn
# This module allows us to select the most appopriate features for our regression
# There exist many different approaches to feature selection, however, we will use one of the simplest

# We will look into: f_regression
# f_regression finds the F-statistics for the *simple* regressions created with each of the independent variables
# In our case, this would mean running a simple linear regression on GPA where SAT is the independent variable
# and a simple linear regression on GPA where Rand 1,2,3 is the indepdent variable
# The limitation of this approach is that it does not take into account the mutual effect of the two features
print(f_regression(x,y))

# There are two output arrays
# The first one contains the F-statistics for each of the regressions
# The second one contains the p-values of these F-statistics
# Since we are more interested in the latter (p-values), we can just take the second array
p_values = f_regression(x,y)[1]
print(p_values)

# To be able to quickly evaluate them, we can round the result to 3 digits after the dot
print(p_values.round(3))

# Creating a summary table

# Let's create a new data frame with the names of the features
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
print(reg_summary)

# Then we create and fill a second column, called 'Coefficients' with the coefficients of the regression
reg_summary ['Coefficients'] = reg.coef_
# Finally, we add the p-values we just calculated
reg_summary ['p-values'] = p_values.round(3)