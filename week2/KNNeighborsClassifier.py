# Deliverable S2
# Classifiers
# KNNeighbors

import time
from sklearn.neighbors import KNeighborsClassifier


def k_nearest_neighbors(x_train, x_test, y_train, y_test, neighbors):
    t1 = time.time()
    # Create a k-NN classifier with 5 neighbors (Creamos el clasificador)
    # Fit the classifier to the data (Entrenar el clasificador)
    # model train metric='euclidean', n_neighbors=1
    knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=neighbors)
    knn_model.fit(x_train, y_train)
    # Se pasan los datos de training y las etiquetas correspondientes.
    t2 = time.time()
    train_time = round(t2 - t1, 3)

    # test prediction
    t3 = time.time()
    knn_model.predict(x_test)
    t4 = time.time()
    prediction_time = round(t4 - t3, 3)

    # Accuracy
    accuracy = 100.0 * knn_model.score(x_test, y_test)

    return accuracy, train_time, prediction_time
