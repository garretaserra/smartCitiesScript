import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes
X = iris.iloc[:, :-1]
y = iris['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Instantiate a Naive Bayes classifier
clf = GaussianNB()

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Compute predicted labels and probabilities
y_pred = clf.predict(X_test)


# Accuracy
print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Compute ROC curve and ROC area for each class
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


accuracy = dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
# Learn to predict each class
for i in range(n_classes):
    clf = GaussianNB()
    clf.fit(X_train, y_train[:, i])
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    accuracy[i] = accuracy_score(y_test[:, i], y_pred)
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob)
    roc_auc[i] = roc_auc_score(y_test[:, i], y_pred_prob)

plt.figure()
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], linestyle='--', label='Random (area = 0.5)')
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='Class %d (area = %0.2f)' % (i, roc_auc[i]))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()