import os
import requests
import pygeohash as pgh
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import concat_ws


# Replace with your actual OpenCage API key
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

def fetch_lat_lon(address):
    """
    Calls the OpenCage Geocoding API to obtain latitude and longitude
    for the given address.
    """
    try:
        # URL-encode the address
        encoded_address = requests.utils.quote(address)
        url = f"https://api.opencagedata.com/geocode/v1/json?q={encoded_address}&key={OPENCAGE_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            print("api.opencagedata.com: success")
            data = response.json()
            if data.get("results"):
                geometry = data["results"][0]["geometry"]
                return (geometry["lat"], geometry["lng"])
    except Exception as e:
        print(f"Error fetching geocode for address '{address}': {e}")
    # Return default coordinates if API call fails or no results found.
    return (0.0, 0.0)

def update_lat(lat, address):
    """
    Returns the latitude if present; otherwise, fetches it via the API.
    """
    if lat is None:
        lat, _ = fetch_lat_lon(address)
        return lat
    return lat

def update_lon(lon, address):
    """
    Returns the longitude if present; otherwise, fetches it via the API.
    """
    if lon is None:
        _, lon = fetch_lat_lon(address)
        return lon
    return lon

def generate_geohash(lat, lon):
    """
    Generates a four-character geohash using the provided latitude and longitude.
    """
    try:
        # Precision of 4 produces a 4-character geohash.
        return pgh.encode(lat, lon, precision=4)
    except Exception as e:
        return None

def main():
    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("RestaurantWeatherETL") \
        .master("local[*]") \
        .getOrCreate()

    restaurants_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("/app/data/restaurant_csv/*.csv")

    weather_df = spark.read.parquet("/app/data/weather/*/*/*/*.parquet")

    # Define UDFs for updating latitude and longitude and for generating geohash
    update_lat_udf = F.udf(update_lat, T.DoubleType())
    update_lon_udf = F.udf(update_lon, T.DoubleType())
    geohash_udf    = F.udf(generate_geohash, T.StringType())

    # create address column for reverse geocoding
    restaurants_df = restaurants_df.withColumn("address", concat_ws(", ", F.col("franchise_name"), F.col("city"), F.col("country")))

    # Update latitude and longitude if missing
    restaurants_updated_df = restaurants_df \
        .withColumn("lat", update_lat_udf(F.col("lat"), F.col("address"))) \
        .withColumn("lng", update_lon_udf(F.col("lng"), F.col("address")))

    # Generate the geohash for restaurant data
    restaurants_geo_df = restaurants_updated_df \
        .withColumn("geohash", geohash_udf(F.col("lat"), F.col("lng")))

    # Cast latitude/longitude as doubles (if necessary) and generate geohash
    weather_geo_df = weather_df \
        .withColumn("lat", F.col("lat").cast(T.DoubleType())) \
        .withColumn("lng", F.col("lng").cast(T.DoubleType())) \
        .withColumn("geohash", geohash_udf(F.col("lat"), F.col("lng")))

    # Drop duplicate rows by geohash to avoid data multiplication during join
    weather_unique_df = weather_geo_df.dropDuplicates(["geohash"])

    # Rename conflicting columns
    restaurants_geo_df = restaurants_geo_df \
        .withColumnRenamed("lat", "restaurant_lat") \
        .withColumnRenamed("lng", "restaurant_lng")
    
    weather_unique_df = weather_unique_df \
        .withColumnRenamed("lat", "weather_lat") \
        .withColumnRenamed("lng", "weather_lng")

    # Left join the restaurant data with weather data using the geohash.
    enriched_df = restaurants_geo_df.join(weather_unique_df, on="geohash", how="left")

    # Write the enriched data as partitioned Parquet files.
    enriched_df.write.mode("overwrite") \
        .partitionBy("geohash") \
        .parquet("/app/data/enriched_output")
    
    enriched_df.show(20)

    spark.stop()

if __name__ == "__main__":
    main()
