# Deliverable S2
# Classifiers
# KNNeighbors

import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def k_nearest_neighbors(x_train, x_test, y_train, y_test, neighbors, X, y):
    t1 = time.time()
    # Create a k-NN classifier with 5 neighbors (Creamos el clasificador)
    # Fit the classifier to the data (Entrenar el clasificador)
    # model train metric='euclidean', n_neighbors=1
    knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=neighbors)
    knn_model.fit(x_train, y_train)
    # Se pasan los datos de training y las etiquetas correspondientes.
    t2 = time.time()
    train_time = round(t2 - t1, 3)

    # Cross Validation
    cv_scores = cross_val_score(knn_model, X, y, cv=5) * 100

    # test prediction
    t3 = time.time()
    y_pred = knn_model.predict(x_test)
    t4 = time.time()
    prediction_time = round(t4 - t3, 3)

    # Accuracy
    accuracy = 100.0 * knn_model.score(x_test, y_test)

    # Confusion Matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('KNNeighbors\n', confusion[0:10, 0:10])

    # Classification Report
    classification = classification_report(y_test, y_pred)
    print(classification)

    return accuracy, train_time, prediction_time, cv_scores
