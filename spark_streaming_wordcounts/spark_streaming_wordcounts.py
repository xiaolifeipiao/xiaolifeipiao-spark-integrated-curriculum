import sys


# =======================================================================================================================
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a',encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("./log.txt", sys.stdout)
sys.stderr = Logger("./log_1.txt", sys.stderr)		# redirect std err, if necessary

# now it works
print('print log')

def showHot(rdd):
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
    # ==================================================================================================================
    # a)	统计每个时间段(自定)每个单词出现的总次数
    pairs = words.map(lambda x: (x, 1))
    counts = pairs.reduceByKey(lambda a, b: a+b)
    counts.pprint()

    # counts = words.countByWindow(10,10)
    # counts.pprint()
    # #
    # windowedWordCounts = pairs.reduceByWindow(lambda x, y: x + y, lambda x, y: x - y, 10, 10)
    # windowedWordCounts.pprint()
    # ==================================================================================================================
    # b)	统计每个时间段，所有单词出现的总次数
    count=counts.map(lambda x:("all_times:",x[1])).reduceByKey(lambda x,y:x+y)
    count.pprint()

    # ==================================================================================================================
    # c)	统计每个时间段的热词
    counts.foreachRDD(lambda r:showHot(r))



    # windowedWordCounts = pairs.reduceByKeyAndWindow(lambda x, y: x + y, lambda x, y: x - y, windowDuration=10, slideDuration=10)
    # windowedWordCounts.pprint()
    #
    # pairs.foreachRDD(lambda r: showWindow(r))
    # foreach(windowedWordCounts)
    # ==================================================================================================================
    print("测试")
    ssc.start()
    ssc.awaitTermination()

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

