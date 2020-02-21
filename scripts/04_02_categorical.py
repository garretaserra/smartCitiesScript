import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
print(iris.dtypes)

#
# Handling categorical data
# 

# Using Pandas df.cat.codes & df.cat.categories
# categories to codes (numerical)
iris['variety'] = iris['variety'].astype('category')
mapping = dict(enumerate(iris['variety'].cat.categories))
iris['variety'] = iris['variety'].cat.codes

# reverse
inv_mapping = {v: k for k, v in mapping.items()}
iris['variety'] = iris['variety'].map(inv_mapping)

# Using map()
# Manual nominal to numerical mapping
mapping = {'Setosa': 0,
       'Versicolor': 1, 
       'Virginica': 2}

iris['variety'] = iris['variety'].map(mapping)

# reverse
inv_mapping = {v: k for k, v in mapping.items()}
iris['variety'] = iris['variety'].map(inv_mapping)



# Using map(): general case
# to convert from strings to integers
mapping = {label: idx for idx, label in enumerate(np.unique(iris['variety']))}
iris['variety'] = iris['variety'].map(mapping)


# Using sklearn.preprocessing.LabelEncoder
le = LabelEncoder()
iris['variety'] = le.fit_transform(iris['variety'].values)

# reverse
iris['variety'] = le.inverse_transform(iris['variety'])


# Other encoders

# One-Hot Encoding: binary matrix
# Using Pandas
iris = pd.get_dummies(iris, columns=['variety'])

# Using sklearn.preprocessing.OneHotEncoder
ohe = make_column_transformer(
    (OneHotEncoder(), ['variety']),
    remainder="passthrough"
)
iris = ohe.fit_transform(iris)


# Binning numerical columns
# Using Pandas
iris['Cat.sepal.length'] = pd.qcut(iris['sepal.length'], q=4, labels=False )

# Using sklearn.preprocessing.KBinsDiscretizer
# https://scikit-learn.org/dev/auto_examples/preprocessing/plot_discretization_strategies.html
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
iris = kbd.fit_transform(iris)
