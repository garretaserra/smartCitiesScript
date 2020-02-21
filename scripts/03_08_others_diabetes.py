import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Load dataset
df = pd.read_csv('../datasets/diabetes.csv')
X = df.drop('diabetes', axis=1).values
y = df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

accuracy = dict()
roc_auc = dict()

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=2,
                              random_state=1)
tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
y_pred_prob = tree.predict_proba(X_test)[:,1]
accuracy['DecisionTreeClassifier'] = accuracy_score(y_test, y_pred)
roc_auc["DecisionTreeClassifier"] = roc_auc_score(y_test, y_pred_prob)


forest = RandomForestClassifier(max_depth=2, 
                                n_estimators=10,
                                random_state=1)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
y_pred_prob = forest.predict_proba(X_test)[:,1]
accuracy['RandomForestClassifier'] = accuracy_score(y_test, y_pred)
roc_auc["RandomForestClassifier"] = roc_auc_score(y_test, y_pred_prob)

# xgboost in Anaconda
# conda install -c anaconda py-xgboost
xgb = XGBClassifier(max_depth=2, 
              n_estimators=10, 
              random_state=1, 
              n_jobs=-1, learning_rate=0.1)
xgb = xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_pred_prob = xgb.predict_proba(X_test)[:,1]
accuracy['XGBoost'] = accuracy_score(y_test, y_pred)
roc_auc["XGBoost"] = roc_auc_score(y_test, y_pred_prob)






