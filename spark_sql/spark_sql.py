#-*- coding: utf-8 -*-
from __future__ import print_function
import findspark

findspark.init()
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("TfIdf Example") \
        .getOrCreate()

    train_file = "E:/学习资料/Spark快速大数据分析/课程设计论文/train.csv"
    data = spark.read.csv(path=train_file, header='true', inferSchema='true')  # 自动推断编码方式

    data = data.select(['DayOfWeek','Category','Descript'])  # 保留Category,Descript,作为数据库

    # data.show(5)

    # data.printSchema() #打印dataFrame的结构
    data.groupBy("DayOfWeek") \
        .count() \
        .orderBy(col("count").desc()) \
        .show() #对Category分组，并计数，以计数结果降序排列输出

    data.groupBy("Category") \
        .count() \
        .orderBy(col("count").desc()) \
        .show() #对Descript分组，并计数，以计数结果降序排列输出

    labelIndexer = StringIndexer().setInputCol("Category").setOutputCol(
        "label")  # 创建StringIndexer对象，设置输入输出参数，按照频率给标签数值化
    data2 = labelIndexer.fit(data).transform(data)




    sc = spark.sparkContext
    rdd_files = sc.wholeTextFiles("E:/学习资料/Spark快速大数据分析/课程设计论文/test")

    schema = StructType([StructField("name", StringType(), True), StructField("content", StringType(), True)])
    df = spark.createDataFrame(rdd_files, schema)

    df.select("*").show()

    tokenizer = Tokenizer(inputCol="content", outputCol="words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    featurizedData.select("rawFeatures").show()
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("name", "features").show()



    # Trains a k-means model.
    kmeans = KMeans().setK(5).setSeed(1)
    model = kmeans.fit(rescaledData)

    # Make predictions
    predictions = model.transform(rescaledData)
    predictions.select('name', 'prediction').show()
    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

spark.stop()
