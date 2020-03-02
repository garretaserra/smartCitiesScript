# Deliverable S2
# Classifiers
# NaiveBayes

import time
from sklearn.naive_bayes import GaussianNB


def naive_bayes(x_train, x_test, y_train, y_test):
    # Train model
    t1 = time.time()
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)
    t2 = time.time()
    train_time = round(t2-t1, 3)
    # print('Time used for training with a size of', len(x_train), 'is', train_time)

    # test prediction
    t1 = time.time()
    y_pred = gnb_model.predict(x_test)
    t2 = time.time()
    prediction_time = round(t2-t1, 3)
    # print('Time for prediction with size', len(x_test), 'is: ', prediction_time)
    # print('Misclassified samples: %d' % (y_test != y_pred).sum())
    # print('Accuracy: %.2f%%' % (100.0 * gnb_model.score(x_test, y_test)))
    accuracy = 100.0 * gnb_model.score(x_test, y_test)

    return accuracy, train_time, prediction_time
