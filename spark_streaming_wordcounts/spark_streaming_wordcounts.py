import sys

def showWindow(rdd):
    for (k, v) in rdd.collect():
        print(k,v)
try:
    from pyspark import SparkContext
    from pyspark.streaming import StreamingContext
    print ("Successfully imported Spark Modules")

    sc = SparkContext(appName="PythonStreamingWordCount")
    ssc = StreamingContext(sc, 1)
    ssc.checkpoint("./spark_checkpoint")

    lines = ssc.textFileStream("./spark_streaming")
    words = lines.flatMap(lambda line: line.split(" "))

    pairs = words.map(lambda x: (x, 1))
    #counts = pairs.reduceByKey(lambda a, b: a+b)
    #counts.pprint()

    print("hello world")
    counts = words.countByWindow(30, 30)
    counts.pprint()

    #windowedWordCounts = pairs.reduceByWindow(lambda x, y: x + y, lambda x, y: x - y, 30, 30)
    #windowedWordCounts.pprint()

    #windowedWordCounts = pairs.reduceByKeyAndWindow(lambda x, y: x + y, lambda x, y: x - y, windowDuration=30, slideDuration=30)
    #windowedWordCounts.pprint()

    pairs.foreachRDD(lambda r: showWindow(r))
    #foreach(windowedWordCounts)

    ssc.start()
    ssc.awaitTermination()

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)