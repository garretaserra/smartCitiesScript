import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv('../datasets/diabetes.csv')
X = df.drop('diabetes', axis=1).values
y = df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

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

# Compute fpr, tpr, thresholds and roc auc
# FPR = FP / (FP + TN)
# TPR = TP / (TP + FN)
y_pred_prob = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label='GaussianNB (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', label='Random (area = 0.5)')
idx = np.abs(thresholds - 0.5).argmin()
plt.plot(fpr[idx], tpr[idx], 'bo', label='threshold=0.5')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


