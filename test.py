import findspark
findspark.init()

from pyspark import SparkContext

sc = SparkContext(appName="PythonCombineByKey")
# 文件路径
movie_path = "D:\\sparkl\\file\\ml-20m\\movies.csv"
ratings_path = "D:\\sparkl\\file\\ml-20m\\ratings.csv"
# 读取电影数据并去除首行
movie_lines = sc.textFile(movie_path)
movie_frist_line = movie_lines.first()
movie_lines = movie_lines.filter(lambda line: line != movie_frist_line)

# 第一题电影的数量
print("问题（1）电影的数量为： ", movie_lines.count())


# 第三题 求取电影的总类别
print("电影的类别情况： ")
import pandas as pd
df = pd.read_csv(movie_path, header=0, encoding="utf-8")
genres_lines = sc.parallelize(df['genres'])

# 使用flatmap统计 genres 的每个类别的个数
tag_lines = genres_lines.flatMap(lambda line:line.split("|")) \
    .map(lambda word: (word,1)).reduceByKey(lambda a, b : a + b)
tag_lines.foreach(print)
print("问题（3）总类数: ", tag_lines.count())



# 第四、五题 求取并显示电影平均评分为5分的电影数、和大于4的电影数
def gen_movie(line, moive_id_index, movie_rating_index):
    parts = line.split(',')
    return (int(parts[moive_id_index]), parts[movie_rating_index])

# 获取 (电影名, 电影id)的rdd
movie_rdd = movie_lines.map(lambda line: gen_movie(line, 0, 1))

# 获取 (电影id, 电影评分)的rdd
rating_lines = sc.textFile(ratings_path)
rating_lines_first = rating_lines.first()
rating_lines = rating_lines.filter(lambda line: line != rating_lines_first)
print("问题（2）评价数: ", rating_lines.count())
rating_rdd = rating_lines.map(lambda line: gen_movie(line, 1, 2))
# 获取评分为5的(电影id， 电影评分)rdd
rating_rdd_5 = rating_rdd.filter(lambda line: float(line[1]) == 5)
# 获取评分大于4的(电影id， 电影评分)rdd
rating_rdd_4 = rating_rdd.filter(lambda line: float(line[1]) >= 4)

# score_rdd_5 = movie_rdd.join(rating_rdd_5)
# score_rdd_4 = movie_rdd.join(rating_rdd_4)

print("问题（4）电影平均评分为5分的电影数",rating_rdd_5.count())
print("问题（5）电影平均评分大于4分的电影数",rating_rdd_4.count())

# score_5_50 = score_rdd_5.take(50)
# score_4_50 = score_rdd_4.take(50)
# print(score_5_50)
# print(score_4_50)


# 第六题
def gen_user_rating(line, user_id_index,movie_rating_index):
    handle_line = line.split(",")
    return (handle_line[user_id_index], [float(handle_line[movie_rating_index]), 1] )

# 生成 (userid,rating,time)的rdd, 最后通过  rating/time 得到平均值
user_rating_rdd = rating_lines.map(lambda line: gen_user_rating(line, 0, 2))
# 计算 rating 和 time
def rd(a,b):
    return [a[0]+b[0], a[1]+b[1]];
user_rating_rdd = user_rating_rdd.reduceByKey(rd)
user_rating_rdd = user_rating_rdd.map(lambda line: (line[0], line[1][0]/line[1][1]))
print("问题（6）用户的平均评分值:", user_rating_rdd.take(10))


# 第7题 每部电影的评价次数
def movie_id(line):
    handle_line = line.split(",")
    return (int(handle_line[1]), 1)
movie_ids = rating_lines.map(movie_id)
movie_ids = movie_ids.reduceByKey(lambda a,b: a + b)
movie_ids = movie_rdd.join(movie_ids)
print("问题（7）每部电影的评价次数", movie_ids.take(10))