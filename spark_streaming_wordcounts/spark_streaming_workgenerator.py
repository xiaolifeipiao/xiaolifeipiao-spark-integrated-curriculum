import random
import time
import shutil
import os
from pyspark import SparkContext

sc = SparkContext('local','spark_streaming_wordcounts')

def my_mkdir(dir_path):
    if os.path.exists(dir_path):
        return dir_path
    else:
        try:
            os.mkdir(dir_path)
        except:
            print(dir_path, " creation fails")
            return -1
        return dir_path

def is_all_whitespace(line):
    for char in line:
        if char != ' ' and char != '\n' and char != '\t':
            return False
    return True

def get_words(file_path):
    r = []
    f = open(file_path,'r')
    for line in f.readlines():
        if is_all_whitespace(line):
            continue
        words = line.split()
        r = r + words
    return r

src_file_path = "input/spark_streaming_dict.txt"
print(get_words(src_file_path))

def generator_word(src_file_path, dst_file_path):
    words = get_words(src_file_path)
    word_count= len(words)
    max_iter = 100000
    save_path = dst_file_path + '_' + str(0)
    f = open(save_path, 'w')
    for i in range(0, max_iter):
        print("第{0}次运行".format(i))
        index = random.randint(0, word_count - 1)
        random_word = words[index]
        f.write(random_word + "  \n")
        if i > 0 and i% 5 == 0:
            f.close()
            shutil.copy(save_path, "./spark_streaming/")
            os.remove(save_path)
            save_path = dst_file_path + '_' + str(i)
            f = open(save_path, 'w')
            time.sleep(5)
    print("折行完毕")
    return

dst_file_path = 'log.txt'
my_mkdir("./spark_streaming/")
generator_word(src_file_path, dst_file_path)