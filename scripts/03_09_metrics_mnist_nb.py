from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

###
### HOLD-OUT EVALUATION
###

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Instantiate a Naive Bayes classifier
clf = GaussianNB()

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
# https://scikit-learn.org/0.15/modules/model_evaluation.html
# "macro": calculate metrics for each label, and find their mean. This does not take label imbalance into account.
# "weighted": calculate metrics for each label, and find their average weighted by the number of occurrences of the label in the true data. This alters "macro" to account for label imbalance; it may produce an F-score that is not between precision and recall.
print(classification_report(y_test, y_pred))

# Accuracy
acc = clf.score(X_test, y_test)
print("Accuracy (hold-out): {0:.2f}".format(acc))


###
### K-FOLD CROSS-VALIDATION
###

# Create a new KNN model
clf_cv = GaussianNB()

# Train model with cv of 5 
cv_scores = cross_val_score(clf_cv, X, y, cv=5)

# Print each cv score (accuracy) 
print("Cross-validation scores: ", cv_scores)

# Print the mean score and the 95% confidence interval of the score estimate
print("Accuracy (5-cv): {0:0.2f} (+/- {1:0.2f})".format(cv_scores.mean(), cv_scores.std() * 2))
