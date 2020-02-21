import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')

# EDA
print(iris.info())
print(iris.groupby('variety').size())
print(iris.describe(include='all'))
sns.pairplot(iris, hue="variety")
sns.lmplot(x='petal.length', y='petal.width', data=iris, hue="variety", fit_reg=False)

