'''
https://fred.stlouisfed.org/series/IPG2211A2N
Monthly Industrial production of electric and gas utilities in the United States, from the years 1985–2018
'''

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

### Load and processing dataset
# - set 'Date' to be the index 
# - convert strings to Datetime
# - rename column
data = pd.read_csv('../datasets/IPG2211A2N.csv', index_col = 0)
data.index = pd.to_datetime(data.index) 
data.columns = ['Energy Production']
data.plot()

for i in range(1,10):
    data["EP lag " + str(i)] = data['Energy Production'].shift(i)
 
# Class to the last column
cols = data.columns.tolist()
n = int(cols.index('Energy Production'))
cols = cols[:n] + cols[n+1:] + [cols[n]]
data = data[cols]
    
X = data.iloc[:, :-1]
y = data['Energy Production']


### Train Test Split
X_train = X.loc['1985-01-01':'2014-12-01']
X_test = X.loc['2015-01-01':]
y_train = y.loc['1985-01-01':'2014-12-01']
y_test = y.loc['2015-01-01':]


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Error Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

print("Mean absolute error : {0:.2f}".format(mean_absolute_error(y_test, y_1)))
print("Mean absolute error : {0:.2f}".format(mean_absolute_error(y_test, y_2)))
print("Explained variance score : {0:.2f}".format(explained_variance_score(y_test, y_1)))
print("Explained variance score : {0:.2f}".format(explained_variance_score(y_test, y_2)))

x_coordinate = [ i for i in  range(len(y_1)) ]

# Plot the results
plt.figure()
plt.scatter(x_coordinate, y_test, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_coordinate, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(x_coordinate, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
