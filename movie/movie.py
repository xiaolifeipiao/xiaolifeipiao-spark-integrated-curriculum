import pandas as pd
from pyspark import SparkContext

# ===============================================================
# 获取列进行遍历返回元组（）
def get_column(line, column_index1,column_index2):
    if 'userId' in line or 'movieId' in line:
        return (-1,-1)
    parts = line.split(',')
    return (int(parts[column_index1]), parts[column_index2])

# 对rating的数据进行过滤，返回对应的值
def filter_rating(line, column_index,threshold):
    if 'userId' in line or 'movieId' in line:
        return False
    parts = line.split(',')
    part = parts[column_index]
    if float(part) >= threshold:
        return True
    else:
        return False

def seq_op(r, rating):
    total_rating = r[0] + float(rating)
    count = r[1] + 1
    return (total_rating, count)

def comb_op(p1,p2):
    return (p1[0]+p2[0],p1[1]+p2[1])

def my_avg(x):
    (key, (total, count)) = x
    return (key,total/count)
# 切割电影类别
def get_genres(line):
    if line==1:
        return (-1,-1)
    parts = line.split('|')
    return (1,parts)
# 我的每部电影评分
def my_num(x):
    (key, (total, count)) = x
    return (key, count)

if __name__ == "__main__":
    sc =SparkContext("local", 'movie')
    # 电影数据路径
    input_path_movies = "input/movies.csv"
    # 电影的评论路径
    input_path_ratings = "input/ratings.csv"
    # 输出路径
    output_path = "output"
    # 定义需要的函数
    # ================================================================================================
    # 题目1：针对电影数据集，进行Spark中的RDD编程，具体要求如下：
    # 1）求取并显示电影的总数量
    # 对应的每一列下标
    movie_id_index_movie = 0
    movie_title_index = 1
    # 读取电影数据
    movie_lines = sc.textFile(input_path_movies)
    movie_rdd = movie_lines.map(lambda x: get_column(x, movie_id_index_movie, movie_title_index))
    movie_count=movie_rdd.count()
    print("电影总数为:",(movie_count-1))
    # 电影总数为：27278
    # ==================================================================================================
    # 2）求取并显示用户评价的总数量
    movie_id_index_rating = 1
    rating_index = 2
    rating_lines = sc.textFile(input_path_ratings)
    rating_rdd = rating_lines.map(lambda x: get_column(x, movie_id_index_rating,rating_index))
    rating_count=rating_rdd.count()
    print("获取的评论总数为：",rating_count)
    # ===================================================================================================
    # 3）	求取并显示电影的总类数
    # 方法一
    df = pd.read_csv(input_path_movies, header=0, encoding="utf-8")
    genres_lines = sc.parallelize(df['genres'])
    # 使用flatmap统计 genres 的每个类别的个数
    movie_type_lines = genres_lines.flatMap(lambda line:line.split("|")).map(lambda word: (word,1)).reduceByKey(lambda a, b : a + b)
    movie_type_lines.foreach(print)
    print("电影总种类数: ", movie_type_lines.count())
    # 方法二
    movie_id_index_genres=2
    movie_genres_rdd=movie_lines.map(lambda x:get_column(x, movie_id_index_movie,movie_id_index_genres))
    movie_genres_rdd2=movie_genres_rdd.map(lambda x:get_genres(x[1])).filter(lambda x:x[0]>0).flatMapValues(lambda x:x)
    print("电影总种类数：{}".format(movie_genres_rdd2.distinct().count()))

    # ====================================================================================================
    # 4）	求取并显示电影平均评分为5分的电影数
    # 5）	求取并显示电影平均评分大于4分的电影数
    threshold4 = 4
    threshold5 = 5
    agg_rating_rdd = rating_rdd.aggregateByKey((0,0),seq_op,comb_op)
    avg_rating_rdd = agg_rating_rdd.map(lambda x: my_avg(x))
    filtered_rating_rdd5 = avg_rating_rdd.filter(lambda x: x[1]== threshold5)
    rating_5_count = filtered_rating_rdd5.count()
    print("电影平均评分等于5分",rating_5_count-1)
    # 测试合并
    movie_5 = movie_rdd.take(5)
    # print(movie_5)
    rating_5 = filtered_rating_rdd5.take(5)
    # print(rating_5)
    result_rdd5 = movie_rdd.join(filtered_rating_rdd5)
    output5 = result_rdd5.collect()[:5]
    print("电影平均分等于5分的数据 ",output5)

    filtered_rating_rdd4 = avg_rating_rdd.filter(lambda x: x[1]>= threshold4)
    rating_4_count = filtered_rating_rdd4.count()
    print("电影平均评分大于等于4分",rating_4_count - 1)
    result_rdd4 = movie_rdd.join(filtered_rating_rdd4)
    output4 = result_rdd4.collect()[:5]
    print("电影平均评分大于等于4分的数据",output4)
    # ==================================================================================================
    # 6）	求取并显示每个用户的平均评分值
    user_id_index_rating=0
    user_rating_rdd = rating_lines.map(lambda x: get_column(x, user_id_index_rating,rating_index))
    agg_user_rating_rdd = user_rating_rdd.aggregateByKey((0,0),seq_op,comb_op)
    avg_user_rating_rdd = agg_user_rating_rdd.map(lambda x: my_avg(x))
    print("每个用户平均评分为:",avg_user_rating_rdd.collect()[:5])
    # =================================================================================================
    # 7）	求取并显示每部电影的评价次数
    number_rating_rdd=agg_rating_rdd.map(lambda x: my_num(x))
    print("每部电影的评价总次数为：{}".format(number_rating_rdd.collect()[:5]))
    # =================================================================================================


