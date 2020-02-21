import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
names = iris.columns.values
iris['variety'] = iris['variety'].cat.codes

X = iris.iloc[:, :-1]
y = iris['variety']

std_scale = StandardScaler().fit(X)
X_std = std_scale.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)



#
# Brute force
#

# Unsupervised Features correlation 
# 

correlated_features = set()
_correlation_matrix = X.corr(method='spearman')
for i in range(len(_correlation_matrix.columns)):
    for j in range(i):
        if abs(_correlation_matrix.iloc[i, j]) > 0.8:
            _colname = _correlation_matrix.columns[i]
            correlated_features.add(_colname)

print("Unsupervised brute force")
print("Strong correlated features")
print(correlated_features)

# To remove selected features
# X = X.drop(labels=correlated_features, axis=1)


# Supervised Feature Importance
# Using mlxtend.evaluate.feature_importance_permutation
# Using sklearn.neighbors.KNeighborsClassifier

# conda install -c conda-forge mlxtend
from mlxtend.evaluate import feature_importance_permutation
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

feature_importances_, _ = feature_importance_permutation(
    predict_method=knn.predict, 
    X=X_test,
    y=y_test,
    metric='accuracy',
    num_rounds=100,
    seed=1)

feature_importances = pd.DataFrame({'feature':X.columns,'importance':np.round(feature_importances_,3)})
feature_importances = feature_importances.sort_values('importance',ascending=False)


print("Supervised brute force")
print(feature_importances)



#
# Embedded
#

# Classifier
# Using sklearn.ensemble.RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=123)
forest.fit(X_std, y);

feature_importances = pd.DataFrame({'feature':X.columns,'importance':np.round(forest.feature_importances_,3)})
feature_importances = feature_importances.sort_values('importance',ascending=False)

print("Embedded (Classifier) approach")
print(feature_importances)


# Regularization (L1 norm)
# Using sklearn.feature_selection.SelectFromModel

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=10000)
sfm = SelectFromModel(estimator=logistic)
sfm.fit(X_std, y)
feature_importances = pd.DataFrame({'feature':X.columns,'importance':sfm.get_support()})

print("Regularization (L1 norm) Embedded approach")
print(feature_importances)

#
# Wrapper
#

# A recursive feature elimination approach
# Using sklearn.feature_selection.RFE
# Using sklearn.neighbors.SVC

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

svc = SVC(kernel="linear", C=1)

rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X_train, y_train)

feature_importances = pd.DataFrame({'feature':X.columns,'importance':np.round(rfe.ranking_,3)})
feature_importances = feature_importances.sort_values('importance',ascending=False)

print("Recursive feature elimination wrapper")
print(feature_importances)

# A forward selection approach
# Using mlxtend.feature_selection.SequentialFeatureSelector
# Using sklearn.neighbors.KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(estimator=knn)

sfs.fit(X_train, y_train)

print("Forward selection Wrapper subsets")
print(sfs.subsets_)


#
# Filter
#

# Unsupervised
# Variance Threshold (low variance)

from sklearn.feature_selection import VarianceThreshold

threshold = 0.5 # variance threshold
sel = VarianceThreshold(threshold=threshold)
sel.fit(X)

# To remove selected features
# X = sel.transform(X)


# Supervised
# Relief 
# conda install -c conda-forge skrebate

from skrebate import ReliefF

fs = ReliefF()
fs.fit(X, y)
    
feature_importances = pd.DataFrame({'feature':X.columns,'importance':np.round(fs.feature_importances_,3)})
feature_importances = feature_importances.sort_values('importance',ascending=False)

