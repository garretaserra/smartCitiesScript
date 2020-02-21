from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
import numpy as np

#sc = SparkContext("local")
sc = SparkContext.getOrCreate()

data = [LabeledPoint(0.0,[10.0, 20.0, 30.0]),
        LabeledPoint(1.0,[1.0, 1.0, 1.0]),
        LabeledPoint(3.0,[2.0, 0.0, 23.0]),
        LabeledPoint(2.0,[3.0, 2.0, 4.0])]

nv = NaiveBayes.train(sc.parallelize(data))

test = np.array([[1, 2, 4],[2, 4, 23]])

res = nv.predict(sc.parallelize(test))
print(res.collect())

sc.stop()
