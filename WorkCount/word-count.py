import os
import shutil

from pyspark import SparkContext

inputpath = 'input'
outputpath = 'output'

sc = SparkContext('local', 'wordcount')

# 读取文件
input = sc.textFile(inputpath)
# 切分单词
words = input.flatMap(lambda line: line.split(' '))
# 转换成键值对并计数
counts = words.map(lambda word: (word, 1)).reduceByKey(lambda x, y: x + y)

# 输出结果
counts.foreach(print)

# 删除输出目录
if os.path.exists(outputpath):
    shutil.rmtree(outputpath, True)

# 将统计结果写入结果文件
counts.saveAsTextFile(outputpath)