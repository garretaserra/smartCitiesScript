import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# Load the iris dataset
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes
X = iris.iloc[:, :-1]
y = iris['variety']

# Instantiate a knn classifier 
knn = KNeighborsClassifier()

# Setup the hyperparameter grid
param_grid = {'n_neighbors': list(range(1, 21)),
              'weights': ['uniform', 'distance']}

# Instantiate the search
knn_cv = GridSearchCV(knn, param_grid, cv = 10)

# Fit the classifier to the data
knn_cv.fit(X, y)

# Print the tuned parameters and score
print("Best parameters: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))

means = knn_cv.cv_results_['mean_test_score']
stds = knn_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, knn_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))