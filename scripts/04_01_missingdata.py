import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor

# Load titanic dataset
titanic = pd.read_csv(r'..\datasets\titanic.csv')
print(titanic.dtypes)

#
# Handling Missing Values
# 

# Identifying missing values
print(titanic.isnull().sum())

# Removing rows that contain missing values
new = titanic.dropna()
new = titanic.dropna(axis=0)

# Removing columns that contain missing values
new = titanic.dropna(axis=1)

# Removing only rows where all columns are NaN
new = titanic.dropna(how='all')

# Removing rows that have less than limit real values 
limit = len(titanic.index) * 0.7 # 70% of nan
new = titanic.dropna(axis=1, thresh=limit)

# Imputing missing values
# Using Pandas
new.loc[:,'Age'] = titanic['Age'].replace(np.nan, titanic['Age'].mean()) 
new.loc[:,'Age'] = titanic['Age'].fillna(titanic['Age'].mean()) 

# Backward fill
new.loc[:,'Age'] = titanic['Age'].fillna(method='bfill')
# Forward fill
new.loc[:,'Age'] = titanic['Age'].fillna(method='ffill')

# Using sklearn.impute.SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
new.loc[:,'Age'] = imputer.fit_transform(new[['Age']])

# Using sklearn.impute.IterativeImputer
# only available in scikit-learn 0.21, released as a developer version, not as stable.
imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
new = pd.DataFrame(imp.fit_transform(titanic), columns=titanic.columns)
