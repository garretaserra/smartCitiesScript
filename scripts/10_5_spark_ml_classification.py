from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import NaiveBayes

#sc = SparkContext("local")
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

data = sqlContext.createDataFrame([
    (0.0, Vectors.dense([10.0, 20.0, 30.0])),
    (1.0, Vectors.dense([1.0, 1.0, 1.0])),
    (3.0, Vectors.dense([2.0, 0.0, 23.0])),
    (2.0, Vectors.dense([3.0, 2.0, 4.0]))], ["label", "features"])

nb = NaiveBayes()
model = nb.fit(data)

test = sqlContext.createDataFrame([
    (0.0, Vectors.dense([1, 2, 4])),
    (1.0, Vectors.dense([2, 4, 23]))], ["label", "features"])

predictions = model.transform(test)
predictions.show()

sc.stop()