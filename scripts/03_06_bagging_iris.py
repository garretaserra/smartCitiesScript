import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Load iris dataset
iris = pd.read_csv("..\datasets\iris.csv")
iris['variety'] = iris['variety'].astype('category')
iris['variety'] = iris['variety'].cat.codes
X = iris.iloc[:, :-1]
y = iris['variety']
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


names = ["DecisionTreeClassifier", "BaggingClassifier"]


accuracy = pd.DataFrame(columns=names)
roc_auc = pd.DataFrame(columns=names)


for i in range(n_classes):
    tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=2,
                              random_state=1)
    tree = tree.fit(X_train, y_train[:, i])
    y_pred = tree.predict(X_test)
    y_pred_prob = tree.predict_proba(X_test)[:,1]
    accuracy.at[i, 'DecisionTreeClassifier'] = accuracy_score(y_test[:, i], y_pred)
    roc_auc.at[i, 'DecisionTreeClassifier'] = roc_auc_score(y_test[:, i], y_pred_prob)

    
for i in range(n_classes):
    bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)
    bag = bag.fit(X_train, y_train[:, i])
    y_pred = bag.predict(X_test)
    y_pred_prob = bag.predict_proba(X_test)[:,1]
    accuracy.at[i, 'BaggingClassifier'] = accuracy_score(y_test[:, i], y_pred)
    roc_auc.at[i, 'BaggingClassifier'] = roc_auc_score(y_test[:, i], y_pred_prob)
