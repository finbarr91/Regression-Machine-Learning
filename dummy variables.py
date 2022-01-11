import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('1.03. Dummies.csv')
print(raw_data.head(50))

data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No' : 0})
print(data.head(10))
print(data.describe())

# Regression
y = data['GPA']
x1 = data[['SAT', 'Attendance']]
x= sm.add_constant(x1)
print(x)
results = sm.OLS(y,x).fit()
print(results.summary())

plt.scatter(data['SAT'],y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw =2, c = '#a50026')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670], 'Attendance' : [0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
print(new_data.head())
new_data.rename(index= {0:'Bob', 1:'Alice'})
predictions = results.predict(new_data)
print(predictions)