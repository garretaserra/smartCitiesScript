# Deliverable S2
# Classifiers
# Support Vector Machine

from sklearn.svm import SVC
import time
from sklearn.model_selection import cross_val_score


def support_vector_machine(x_train, x_test, y_train, y_test, X, y):
    # Train model
    t1 = time.time()
    svm_model = SVC(kernel='linear')
    svm_model.fit(x_train, y_train)
    t2 = time.time()
    train_time = round(t2-t1, 3)

    cv_scores = cross_val_score(svm_model, X, y, cv=5) * 100

    # test prediction
    t1 = time.time()
    svm_model.predict(x_test)
    t2 = time.time()
    prediction_time = round(t2-t1, 3)
    accuracy = 100.0 * svm_model.score(x_test, y_test)
    return accuracy, train_time, prediction_time, cv_scores