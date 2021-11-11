import math

import pandas
import pandas as pd
from pyspark import SparkContext
sc =SparkContext("local", 'movie')
# 电影数据路径
input_path_movies = "input/movies.csv"
# 电影的评论路径
input_path_ratings = "input/ratings.csv"
output_path = "output"
# 题目1：针对电影数据集，进行Spark中的RDD编程，具体要求如下：
# 1）求取并显示电影的总数量
# movie_lines = sc.textFile(input_path_movies)
# print("获取的电影总数为：",movie_lines.count())
# 2）求取并显示用户评价的总数量
# ratings_lines =sc.textFile(input_path_ratings)
# print("获取的评论总数为：",ratings_lines.count())
# 3）	求取并显示电影的总类数

# 4）	求取并显示电影平均评分为5分的电影数

def get_column(line, column_index1,column_index2):
    if 'userId' in line or 'movieId' in line:
        return (-1,-1)
    parts = line.split(',')
    return (int(parts[column_index1]), parts[column_index2])

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

if __name__ == "__main__":
    movie_id_index_movie = 0; movie_title_index = 1
    movie_lines = sc.textFile(input_path_movies)
    movie_rdd = movie_lines.map(lambda x: get_column(x,movie_id_index_movie, movie_title_index))

    movie_id_index_rating = 1;rating_index = 2
    threshold4 = 4
    threshold5 = 5
    rating_lines = sc.textFile(input_path_ratings)
    rating_rdd = rating_lines.map(lambda x: get_column(x, movie_id_index_rating,rating_index))
    agg_rating_rdd = rating_rdd.aggregateByKey((0,0),seq_op,comb_op)
    avg_rating_rdd = agg_rating_rdd.map(lambda x: my_avg(x))
    filtered_rating_rdd = avg_rating_rdd.filter(lambda x: x[1]== threshold5)

    # f6……#

    movie_5 = movie_rdd.take(5)
    # print(movie_5)

    rating_5 = filtered_rating_rdd.take(5)
    # print(rating_5)
    result_rdd5 = movie_rdd.join(filtered_rating_rdd)

    output5 = result_rdd5.collect()[:5]
    print("等于5分",output5)
    # for (movie_id, l) in output:
    #     print(movie_id, end ='\t')
    #     print(l)
# 5）	求取并显示电影平均评分大于4分的电影数
    filtered_rating_rdd4 = avg_rating_rdd.filter(lambda x: x[1]>= threshold4)
    result_rdd4 = movie_rdd.join(filtered_rating_rdd)
    output4 = result_rdd4.collect()[:5]
    print("大于等于4分",output4)
# 6）	求取并显示每个用户的平均评分值
    userid_rating_rdd = rating_lines.map(lambda x: get_column(x, 0,2))
    agg_userid_rating_rdd = userid_rating_rdd.aggregateByKey((0,0),seq_op,comb_op)
    avg_userid_rating_rdd = agg_userid_rating_rdd.map(lambda x: my_avg(x))
    print(avg_userid_rating_rdd.take(5))
# 7）	求取并显示每部电影的评价次数



