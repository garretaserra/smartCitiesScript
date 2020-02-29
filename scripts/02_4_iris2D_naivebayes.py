import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
from util import plot_decision_regions

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes

# Split in train and test datasets
# 2D Attributes
X = iris[['petal.length','petal.width']]
y = iris['variety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Class labels:', np.unique(y))
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))



# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# 
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)


# test prediction
y_pred = gnb_model.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f%%' % (100.0 * gnb_model.score(X_test, y_test)))


# decision boundary
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
train_len = X_train.shape[0]
combined_len = X_combined.shape[0]

plt.figure(figsize=(3, 3), dpi=300)
plot_decision_regions(X=X_combined, y=y_combined, classifier=gnb_model, test_idx=range(train_len, combined_len))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()