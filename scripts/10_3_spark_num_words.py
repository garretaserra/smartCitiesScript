from pyspark import SparkContext

#sc = SparkContext("local")
sc = SparkContext.getOrCreate()

# Creating a RDD from a file
distFile=sc.textFile("../Datasets/hamlet.txt")

# Processing data with map&reduce
num_words = distFile \
.flatMap(lambda line: line.split()) \
.filter(lambda word: word != '') \
.map(lambda word: 1) \
.reduce(lambda a, b: a+b)

print(num_words)

sc.stop()
