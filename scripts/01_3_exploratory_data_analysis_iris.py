import pandas as pd


# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")

# Clase: categorical to numerical
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes


##### Attribute

# Attribute information
print(iris.info())

# Attributes distributions
print(iris.describe(include='all'))

# Attribute distribution
print(iris['sepal.length'].value_counts().sort_index())


##### Attribute to Attribute relationship

# Correlation: correlation coefficient matrix
iris_corr = iris.corr()
print(iris_corr)

# Correlation coefficient: feature vs feature
iris_corr_values = iris_corr['variety'][:-1]
iris_corr_values = iris_corr_values.sort_values(ascending=False)
print(iris_corr_values)





