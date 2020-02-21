import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load creditcard_small dataset
creditcard = pd.read_csv(r'..\datasets\creditcard_small.csv')
creditcard = creditcard.drop(['Time'], axis = 1)

X = creditcard.iloc[:,:-1]
y = creditcard.iloc[:,-1]


print('Class labels:', np.unique(y))
print('Labels counts in y:', np.bincount(y))

sns.scatterplot(X.V4, X.V6, y, alpha=.5, legend=False)

# Resampling Imbalanced Data
# conda install -c conda-forge imbalanced-learn
# https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
# not majority': resample all classes but the majority class
# 'minority': resample only the minority class
# 'not majority': resample all classes but the majority class
# 'all': resample all classes
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='minority', random_state = 33)
X_resampled, y_resampled = sm.fit_sample(X, y)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

print('Class labels:', np.unique(y_resampled))
print('Labels counts in y_resampled:', np.bincount(y_resampled))

sns.scatterplot(X_resampled.V4, X_resampled.V6, y_resampled, alpha=.5, legend=False)


# Impact on the prediction
# standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2, shuffle = True, stratify = y)

# import logistic regression model and accuracy_score metric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = LogisticRegression(solver = 'lbfgs')

# Without resampling Imbalanced Data
print('Without resampling Imbalanced Data')
clf.fit(X_train, y_train.ravel())
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, y_test))
print('Confusion Matrix - Testing Dataset')
print(pd.crosstab(y_test.ravel(), test_pred.ravel(), rownames = ['True'], colnames = ['Predicted'], margins = True))

# With resampling Imbalanced Data
print('With resampling Imbalanced Data')
sm = SMOTE(sampling_strategy='minority', random_state = 33)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())
clf.fit(X_train_new, y_train_new)
train_pred_sm = clf.predict(X_train_new)
test_pred_sm = clf.predict(X_test)
print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred_sm, y_test))
print('Confusion Matrix - Testing Dataset')
print(pd.crosstab(y_test.ravel(), test_pred_sm, rownames = ['True'], colnames = ['Predicted'], margins = True))

