from pyspark import SparkContext
import time
import matplotlib.pyplot as plt

from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql import SQLContext

from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.feature import HashingTF,StopWordsRemover,IDF,Tokenizer


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
print('print something')
# =======================================================================================================================

sc = SparkContext("local","Application")
sqlContext = SQLContext(sc)
path="input/test\\*"
start_time = time.time()
newsGroupRowData=sc.wholeTextFiles(path)
print("文件数量:",newsGroupRowData.count())
end_time = time.time()
dtime = end_time - start_time
print("文件读取==============================================================================================>")
print("读取文件耗时",round(dtime,2))
print("展示文件==============================================================================================>")
print(newsGroupRowData.takeSample(False, 1, 10))
# 路径
filepaths = newsGroupRowData.map(lambda x: x[0])
print("处理路径==============================================================================================>")
print(filepaths.takeSample(False,5, 10))
# 文本
text = newsGroupRowData.map(lambda x: x[1])
print("处理文本==============================================================================================>")
print(text.takeSample(False,1, 10))
# id
id = filepaths.map(lambda filepath: filepath.split("/")[-1])
print("处理id==============================================================================================>")
print(id.take(5))
#主题
topics = filepaths.map(lambda filepath: filepath.split("/")[-2])
print("处理主题==============================================================================================>")
print("未去重",topics.take(20))
print("去重",topics.distinct().take(20))

# 这个模型是在字符中
schemaString = "id text topic"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

# 将模式应用于RDD(filepath,text)=>
newsgroups = newsGroupRowData.map(lambda x: (x[0].split("/")[-1],x[1],x[0].split("/")[-2]))
df = sqlContext.createDataFrame(newsgroups, schema)

# 打印模型
print("打印模型==============================================================================================>")
df.printSchema()
# 使用DataFrame创建临时视图
df.createOrReplaceTempView("newsgroups")
# SQL可以在已注册为表的dataframe上运行
results = sqlContext.sql("SELECT id,topic,text FROM newsgroups limit 5")
print("查询数据==============================================================================================>")
results.show(5)
print("查询，去重，统计，分组，降序===========================================================================>")
results = sqlContext.sql("select distinct topic, count(*) as cnt from newsgroups group by topic order by cnt desc limit 5")
results.show(5
             )
print("统计comp==============================================================================================>")
result_list = df[df.topic.like("comp%")].collect()
new_df = sc.parallelize(result_list).toDF()
new_df.dropDuplicates().show()
print("创建标签==============================================================================================>")
labeledNewsGroups = df.withColumn("label",df.topic.like("comp%").cast("double"))
labeledNewsGroups.sample(False,0.003,10).show(5)
# 测试数据
labeledNewsGroups.sample(False,0.3,10).show(5)


# 测试，训练确定
print("测试，训练确定========================================================================================>")
train_set, test_set = labeledNewsGroups.randomSplit([0.8, 0.2], 12345)
print("总共数据:",labeledNewsGroups.count())
print("训练数据:",train_set.count())
print("测试数据",test_set.count())

# 将字符串列转换成小写并按空格切分
tokenizer = Tokenizer().setInputCol("text").setOutputCol("words")
# 移除停顿词
remover= StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(False)
# 词频统计
hashingTF = HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
# 文本tf-idf计算
idf = IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
#


# Logistic回归
lr = LogisticRegression().setRegParam(0.01).setThreshold(0.5)
# lr = KMeans().setK(5).setSeed(1)
pipeline=Pipeline(stages=[tokenizer,remover,hashingTF,idf, lr])
print("Logistic回归========================================================================================>")
print("Logistic回归特征列=",lr.getFeaturesCol())
print("Logistic回归标签列=",lr.getLabelCol())
print("Logistic回归的阈值=",lr.getThreshold())
print("=====================================================================================================")
print("Tokenizer:",tokenizer.explainParams())
print("=====================================================================================================")
print("Remover",remover.explainParams())
print("=====================================================================================================")
print("HashTF",hashingTF.explainParams())
print("=====================================================================================================")
print("IDF:",idf.explainParams())
print("=====================================================================================================")
print("LogisticRegression:",lr.explainParams())
print("=====================================================================================================")
print("Pipeline:",pipeline.explainParams())
print("=====================================================================================================")

print("====================================================================================================>")
print("去除的常用单词",remover.getStopWords())

# 模型预测
print("模型预测=============================================================================================>")
model=pipeline.fit(train_set)
predictions = model.transform(test_set)
predictions.select("id","topic","probability","prediction","label").sample(False,0.01,10).show(5)
predictions.select("id","topic","probability","prediction","label").filter(predictions.topic.like("comp%")).sample(False,0.1,10).show(5)
predictions.sample(False,0.01,10).show(5)


# ROC曲线,模型评估
evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC")
print("Area under ROC curve:",evaluator.evaluate(predictions))

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures,[1000,10000,100000]) \
    .addGrid(idf.minDocFreq,[0,10,100]) \
    .build()
cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)
cvModel = cv.fit(train_set)
print("Area under the ROC curve for best fitted model =",evaluator.evaluate(cvModel.transform(test_set)))
print("Area under ROC curve for non-tuned model:",evaluator.evaluate(predictions))
print("Area under ROC curve for fitted model:",evaluator.evaluate(cvModel.transform(test_set)))
# print("Improvement:%.2f".format(evaluator.evaluate(cvModel.transform(test_set)) - evaluator.evaluate(predictions))*100 / evaluator.evaluate(predictions))


# kmeans
km = KMeans(k=3).setK(5).setSeed(1)

km_pipeline=Pipeline(stages=[tokenizer,remover,hashingTF,idf, km])
km_model=km_pipeline.fit(train_set)
km_predictions = km_model.transform(test_set)
km_predictions.select('label',"features").sample(False,0.01,10).show(5)
# km_predictions.select("id","topic","prediction","label").sample(False,0.01,10).show(5)
# km_predictions.select("id","topic","prediction","label").filter(km_predictions.topic.like("comp%")).sample(False,0.1,10).show(5)
km_predictions.sample(False,0.01,10).show(5)
# Evaluate clustering by computing Silhouette score
# 模型评估
km_evaluator = ClusteringEvaluator(predictionCol="prediction")

silhouette = km_evaluator.evaluate(km_predictions)
print("欧几里得距离的平方 = " + str(silhouette))

km_aoc_evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC")
print("Area under ROC curve:",km_aoc_evaluator.evaluate(km_predictions))

# Shows the result.
# centers = km_model.
# print("Cluster Centers: ")
# for center in centers:
#     print(center)
# km_evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC")
# print("Area under ROC curve:",km_evaluator.evaluate(km_predictions))
# km_evaluator = ClusteringEvaluator()
# silhouette = km_evaluator.evaluate(km_predictions)
# print("Silhouette with squared euclidean distance = " + str(silhouette))
# # Shows the result.
# centers = km_model.clusterCenters()
# print("Cluster Centers: ")
# for center in centers:
#     print(center)


# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method showing the optimal K')
# plt.xlabel('K - Number of clusters')
# plt.ylabel('WCSS')
# plt.show()