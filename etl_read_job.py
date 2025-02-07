from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LocalStorageETLJob") \
    .master("local[*]") \
    .getOrCreate()

restaurants_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/app/data/restaurant_csv/*.csv")

restaurants_df.show(20)

weather_df = spark.read.parquet("/app/data/weather/*/*/*/*.parquet")
weather_df.printSchema()

weather_df.show(20)

# Stop the SparkSession
spark.stop()
