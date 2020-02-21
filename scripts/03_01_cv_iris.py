import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
X = iris.iloc[:, :-1]
y = iris['variety']

# Instantiate classifiers
names = ["Nearest Neighbors", "Decision Tree", "Naive Bayes", "Linear SVM", "RBF SVM", "Neural Net"]
classifiers = [
    KNeighborsClassifier(metric='minkowski', n_neighbors=5),
    DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1),
    GaussianNB(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(alpha=1, max_iter=1000)]



# Cross-validation
# iterate over classifiers
cv_accuracy = dict()
for name, clf in zip(names, classifiers):
    # One metric: score (accuracy)
    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=10)
    cv_accuracy[name] = cv_scores.mean()
    print(name)
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))


# Holdout
# iterate over classifiers
split_accuracy = dict()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    split_score = clf.score(X_test, y_test)
    split_accuracy[name] = split_score
    
    # Print the score 
    print(name)
    print("Accuracy: %0.2f" % (split_score))
    
    