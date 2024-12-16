import os, sys
for root, dirs, files in os.walk(f"{os.environ['SPARK_HOME']}/python/lib"):
    for file in files:
        if "zip" in file:
            sys.path.insert(0, os.path.join(root, file))

from pyspark.sql import SparkSession
from onetl.connection import SparkHDFS
from onetl.file import FileDFReader
from onetl.db import DBWriter
from onetl.file.format import CSV
from onetl.connection import Hive
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, year, month, avg, monotonically_increasing_id, regexp_replace

from prefect import flow, task


@task
def create_session():
    # Building spark session for Hive
    # spark = SparkSession.builder \
    # .master("yarn") \
    # .appName("spark-with-yarn") \
    # .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    # .config("spark.hive.metastore.uris", "thrift://team-2-nn:5433") \
    # .enableHiveSupport() \
    # .getOrCreate()
    # Building spark session for HDFS
    spark = SparkSession.builder \
    .master("yarn") \
    .appName("spark-with-yarn") \
    .getOrCreate()
    return spark


@task
def extract_data(spark):
    hdfs = SparkHDFS(host="team-2-nn", port=9000, spark=spark, cluster="test")
    hdfs.check()
    reader = FileDFReader(connection=hdfs, format=CSV(delimiter=";", header=True), source_path="/input")
    df = reader.run(["sample_data.csv"])
    return df


@task
def transform_data(_df):
    df = _df.toDF(*[column.replace('\r', '') for column in _df.columns])
    df = df.select([regexp_replace(col(column), r'\r', '').alias(column) for column in df.columns])

    # 1. Создание нового столбца разницы между ценами Газпрома и Сбербанка
    df = df.withColumn(
        "A_vs_B", col("price_A") - col("price_B")
    )

    # 2. Фильтрация строк, где цена Газпрома больше 190
    df = df.withColumn(
        "price_A_above_200", col("price_A") > 200
    )

    # 3. Добавление индекса (порядкового номера) для каждой строки
    df = df.withColumn(
        "Row_Index", monotonically_increasing_id()
    )

    # 4. Доходность для каждой акции
    df = df.withColumn(
        "return_A", 
        (col("price_A") - F.lag("Price_A").over(Window.orderBy("Date"))) / F.lag("Price_A").over(Window.orderBy("Date"))
    ).withColumn(
        "return_B", 
        (col("Price_B") - F.lag("Price_B").over(Window.orderBy("Date"))) / F.lag("Price_B").over(Window.orderBy("Date"))
    ).withColumn(
        "return_C", 
        (col("Price_C") - F.lag("Price_C").over(Window.orderBy("Date"))) / F.lag("Price_C").over(Window.orderBy("Date"))
    ).withColumn(
        "return_D", 
        (col("Price_D") - F.lag("Price_D").over(Window.orderBy("Date"))) / F.lag("Price_D").over(Window.orderBy("Date"))
    ).withColumn(
        "return_E", 
        (col("Price_E") - F.lag("Price_E").over(Window.orderBy("Date"))) / F.lag("Price_E").over(Window.orderBy("Date"))
    )

    # 5. Скользящее среднее (с окном в 10 дней) для каждой акции
    window_spec_10 = Window.orderBy("Date").rowsBetween(-9, 0)  # Окно 10 дней (текущий день и 9 предыдущих)

    df = df.withColumn(
        "price_A_MA_10", avg("Price_A").over(window_spec_10)
    ).withColumn(
        "price_B_MA_10", avg("Price_B").over(window_spec_10)
    ).withColumn(
        "price_C_MA_10", avg("Price_C").over(window_spec_10)
    ).withColumn(
        "price_D_MA_10", avg("Price_D").over(window_spec_10)
    ).withColumn(
        "price_E_MA_10", avg("Price_E").over(window_spec_10)
    )
    df = df.repartition(4)

    return df


@task
def load_data(df, spark):
    # Saving the DataFrame to Hive 
    # hive = Hive(spark=spark, cluster='test')
    # writer = DBWriter(connection=hive, table="test.stocks", options={"if_exists": "replace_entire_table", "partition_by": "Row_Index"})
    # writer.run(df)

    # Saving the DataFrame to HDFS in partitions
    output_path = "/output/processed_data"
    df.write \
    .mode("overwrite") \
    .format("parquet") \
    .save(output_path)


@flow
def process_data():
    spark_sess = create_session()
    print('SESSION CREATED')
    edata = extract_data(spark=spark_sess)
    print('DATA EXTRACTED')
    tdata = transform_data(_df=edata)
    print('DATA TRANSFORMED')
    load_data(df=tdata, spark=spark_sess)


if __name__ == "__main__":
    process_data.serve(
        name="first-deployment",
        cron="*/10 * * * *",
        description="Processes data from hdfs.",
    )