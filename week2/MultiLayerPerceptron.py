# Deliverable S2
# Classifiers
# MultiLayerPerceptron

from sklearn.neural_network import MLPClassifier
import time


def multi_layer_perceptron(x_train, x_test, y_train, y_test):
    # Train model
    t1 = time.time()
    neural_network_model = MLPClassifier(max_iter=1000)
    neural_network_model.fit(x_train, y_train)
    t2 = time.time()
    training_time = round(t2-t1, 3)

    # test prediction
    t1 = time.time()
    neural_network_model.predict(x_test)
    t2 = time.time()
    prediction_time = round(t2-t1, 3)
    accuracy = 100.0 * neural_network_model.score(x_test, y_test)
    return accuracy, training_time, prediction_time
