import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('../datasets/diabetes.csv')
X = df.drop('diabetes', axis=1).values
y = df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)


accuracy = dict()
roc_auc = dict()

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3,
                              random_state=1)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500, 
                         learning_rate=0.1,
                         random_state=1)

tree = tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)
y_pred_prob = tree.predict_proba(X_test)[:,1]

accuracy["DecisionTreeClassifier"] = accuracy_score(y_test, y_test_pred)
roc_auc["DecisionTreeClassifier"] = roc_auc_score(y_test, y_pred_prob)
print(classification_report(y_test, y_test_pred))

ada = ada.fit(X_train, y_train)
y_test_pred = ada.predict(X_test)
y_pred_prob = ada.predict_proba(X_test)[:,1]

accuracy["AdaBoostClassifier"] = accuracy_score(y_test, y_test_pred)
roc_auc["AdaBoostClassifier"] = roc_auc_score(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))
