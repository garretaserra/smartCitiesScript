import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
# Only categorical attributes
iris_cat = iris.select_dtypes(include=['category'])
iris['variety'] = iris['variety'].cat.codes

# Attribute information
print(iris.info())

# Feature distribution plots
iris.hist(figsize=(16, 20))

# The bi-attribute distribution plots 
sns.pairplot(iris)

for i in range(0, len(iris.columns), 5):
    sns.pairplot(data=iris,
                 x_vars=iris.columns[i:i + 5],
                 y_vars=['variety'])

# Correlation: correlation matrix
f, ax = plt.subplots(figsize=(10, 8))
iris_corr = iris.corr()
sns.heatmap(iris_corr[(iris_corr >= 0.5) | (iris_corr <= -0.4)],
            vmax=1.0, vmin=-1.0,
            annot=True)

# For categorical attributes: frequency distributions
plt.subplots(figsize=(10, 8))
sns.countplot(data=iris_cat, x='variety')

plt.show()