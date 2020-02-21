import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes
X = iris.iloc[:, :-1]
y = iris['variety']

clf = GaussianNB()
clf = SVC(gamma='scale', random_state=0)

# Cross-validation metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html
score = cross_val_score(clf, X, y, scoring='f1_weighted', cv=10)

# multiple metric evaluation
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
results = cross_validate(clf, X, y, scoring=scoring, cv=10)


