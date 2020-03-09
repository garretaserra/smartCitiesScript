# Deliverable S2
# Classifiers
# DecisionTreeClassifier

import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def decision_tree_classifier(x_train, x_test, y_train, y_test, max_depth, X, y):
    t1 = time.time()
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=1)
    tree_model.fit(x_train, y_train)
    t2 = time.time()
    training_time = round(t2 - t1, 3)

    # Cross Validation
    cv_scores = cross_val_score(tree_model, X, y, cv=5) * 100

    # test prediction
    t1 = time.time()
    y_pred = tree_model.predict(x_test)
    t2 = time.time()
    prediction_time = round(t2 - t1, 3)
    accuracy = 100.0 * tree_model.score(x_test, y_test)

    # Confusion Matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Decision Tree\n', confusion[0:10, 0:10])

    # Classification Report
    classification = classification_report(y_test, y_pred)
    print(classification)

    y_pred_prob = tree_model.predict_proba(x_test)
    roc_curve = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print('ROC: ', roc_curve)

    return accuracy, training_time, prediction_time, tree_model.get_depth(), cv_scores, roc_curve
