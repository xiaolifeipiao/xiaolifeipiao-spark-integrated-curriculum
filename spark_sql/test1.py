from pyspark.sql import SparkSession
from pyspark.sql import Row

def basic_datasource_example(spark):
    return False

def generic_file_source_options_example(spark):
    # spark.sql("set spark.sql.files.ignoreCorruptFiles=true")
    # test_corropt_df = spark.read.parquet("./input/20news-bydate-train/rec.autos/")
    # test_corropt_df.show()
    return False

def text_dataset_example(spark):
    path = "./input/20news-bydate-test/alt.atheism/"
    df = spark.read.format("txt").load(path)
    df.show()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Python Spark SQL data source example").getOrCreate()

    basic_datasource_example(spark)
    generic_file_source_options_example(spark)
    # parquet_example(spark)
    # parquet_schema_merging_example(spark)
    # json_dataset_example(spark)
    # csv_dataset_example(spark)
    # text_dataset_example(spark)
    # jdbc_dataset_example(spark)