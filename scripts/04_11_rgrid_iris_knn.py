from scipy.stats import randint
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Instantiate a knn classifier 
knn = KNeighborsClassifier()

# Setup the parameters and distributions to sample
param_dist = {'n_neighbors': randint(1, 31),
              'weights': ['uniform', 'distance']}

# Instantiate the randomized search
knn_cv = RandomizedSearchCV(knn, param_dist, cv = 10)

# Fit the classifier to the data
knn_cv.fit(X, y)

# Print the tuned parameters and score
print("Best parameters: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


means = knn_cv.cv_results_['mean_test_score']
stds = knn_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, knn_cv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

