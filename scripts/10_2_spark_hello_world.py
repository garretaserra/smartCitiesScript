from pyspark import SparkContext

#sc = SparkContext("local")
sc = SparkContext.getOrCreate()

data = sc.parallelize(list("Hello World"))
counts = data \
.map(lambda x: (x, 1)) \
.reduceByKey(lambda a, b: a+b) \
.sortBy(lambda x: x[1], ascending=False) \
.collect()

for (letter, count) in counts:
    print("{}: {}".format(letter, count))

sc.stop()
