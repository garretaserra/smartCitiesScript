from pyspark import SparkContext

#sc = SparkContext("local")
sc = SparkContext.getOrCreate()

# Creating a RDD from a data collection
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)

# Processing data with map&reduce
res = distData.reduce(lambda a, b: a + b)
print(res)

sc.stop()
