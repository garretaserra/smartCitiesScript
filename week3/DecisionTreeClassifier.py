# Deliverable S2
# Classifiers
# DecisionTreeClassifier

import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def decision_tree_classifier(x_train, x_test, y_train, y_test, max_depth, X, y):
    t1 = time.time()
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=1)
    tree_model.fit(x_train, y_train)
    t2 = time.time()
    training_time = round(t2 - t1, 3)

    cv_scores = cross_val_score(tree_model, X, y, cv=5) * 100

    # test prediction
    t1 = time.time()
    tree_model.predict(x_test)
    t2 = time.time()
    prediction_time = round(t2 - t1, 3)
    accuracy = 100.0 * tree_model.score(x_test, y_test)

    return accuracy, training_time, prediction_time, tree_model.get_depth(), cv_scores