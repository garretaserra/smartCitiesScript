'''
https://fred.stlouisfed.org/series/IPG2211A2N
Monthly Industrial production of electric and gas utilities in the United States, from the years 1985–2018
'''

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

### Load and processing dataset
# - set 'Date' to be the index 
# - convert strings to Datetime
# - rename column
data = pd.read_csv('../datasets/IPG2211A2N.csv', index_col = 0)
data.index = pd.to_datetime(data.index) 
data.columns = ['Energy Production']
data.plot()

### Decomposition
result = seasonal_decompose(data, model='multiplicative')
result.plot()

### Train Test Split
train = data.loc['1985-01-01':'2014-12-01']
test = data.loc['2015-01-01':]

### Fit the model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(2, 1, 2, 12))
model_fit = model.fit(disp=False)
print(model_fit.summary())
print("The AIC is: ", model_fit.aic)

### Evaluation
future_forecast = model_fit.predict(start='2015-01-01', end='2019-08-01')

# Error Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

print("Mean absolute error : {0:.2f}".format(mean_absolute_error(test, future_forecast)))
print("Explained variance score : {0:.2f}".format(explained_variance_score(test, future_forecast)))


future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
pd.concat([test,future_forecast],axis=1).plot()


