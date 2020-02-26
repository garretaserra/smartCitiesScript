import time  # Importar el tiempo
from sklearn import datasets  # Clasificadores en SK-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target  # Etiqueta en la última columna

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
# Partir, el número de muestras, de las 150 iniciales unos se utilizaran para test (30%) y  otros para entrenar
# Todos estos algoritmos llevan una componente aleatoria.
# random_state=21 (semilla por la cual empieza la secuencia) y para que haga siempre la misma partición.
# stratify=y va intentar que quede el mismo porcentaje de muestras de cada clase

# Create a k-NN classifier with 5 neighbors (Creamos el clasificador)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the data (Entrenar el clasificador)
knn.fit(X_train, y_train)  # Se pasan los datos de training y las etiquetas correspondientes.
# Con el import time
# t1=time.time()
# t3=time.time()
# t12=round(t2-t1,3)


# Accuracy
acc = knn.score(X_test, y_test)
print("Accuracy: {0:.2f}".format(acc))

# Predict the labels for the test data
y_pred = knn.predict(X_test)
print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))

# Predict the label of a new sample
X_new = [[3.5, 1.2, 4.5, 2.3]]
y_pred = knn.predict(X_new)
print("New sample: {}".format(X_new))
print("New sample prediction: {} {}".format(y_pred, iris.target_names[y_pred]))
