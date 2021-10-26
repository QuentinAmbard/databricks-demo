# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# DBTITLE 1,Package imports
from pyspark.sql.functions import rand, input_file_name, from_json, col
from pyspark.sql.types import *
 
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

#ML import
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir
import mlflow.spark
import mlflow
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep
import re

# COMMAND ----------

# DBTITLE 1,Mount S3 bucket containing sensor data
aws_bucket_name = "quentin-demo-resources"
mount_name = "quentin-demo-resources"

try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  print("bucket isn't mounted, mounting the demo bucket under %s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)

# COMMAND ----------

# DBTITLE 1,Create User-Specific database
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
print("Created variables:")
print("current_user: {}".format(current_user))
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/demo".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("path (default path): {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/global_demo/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))
print("dbName (using database): {}".format(dbName))

# COMMAND ----------

# DBTITLE 1,Reset tables in user's database
tables = ["turbine_bronze", "turbine_silver", "turbine_gold", "turbine_power", "turbine_schema_evolution"]
reset_all = dbutils.widgets.get("reset_all_data") == "true" or any([not spark.catalog._jcatalog.tableExists(table) for table in ["turbine_power"]])
if reset_all:
  print("resetting data")
  for table in tables:
    spark.sql("""drop table if exists {}.{}""".format(dbName, table))
    
#   spark.sql("""drop database if exists {} CASCADE""".format(dbName))
  spark.sql("""create database if not exists {} LOCATION '{}/tables' """.format(dbName, path))
  dbutils.fs.rm(path+"/turbine/bronze/", True)
  dbutils.fs.rm(path+"/turbine/silver/", True)
  dbutils.fs.rm(path+"/turbine/gold/", True)
  dbutils.fs.rm(path+"/turbine/_checkpoint", True)
      
  spark.read.format("json") \
            .schema("turbine_id bigint, date timestamp, power float, wind_speed float, theoretical_power_curve float, wind_direction float") \
            .load("/mnt/quentin-demo-resources/turbine/power/raw") \
       .write.format("delta").mode("overwrite").save(path+"/turbine/power/bronze/data")
  
  spark.sql("create table if not exists turbine_power using delta location '"+path+"/turbine/power/bronze/data'")
  
  # Create Gold Table containing "Labels"
  spark.sql("create table if not exists turbine_status_gold (id int, status string) using delta")
  spark.sql("""
  COPY INTO turbine_status_gold
    FROM '/mnt/quentin-demo-resources/turbine/status'
    FILEFORMAT = PARQUET;
  """)
  
else:
  print("loaded without data reset")

  
# Define the default checkpoint location to avoid managing that per stream and making it easier. In production it can be better to set the location at a stream level.
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/turbine/_checkpoint")

#Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")

# COMMAND ----------

# DBTITLE 1,Create "gold" tables for autoML(remove ID/Timestamp columns) and ML purposes
if reset_all:
  dataset = spark.read.load("/mnt/quentin-demo-resources/turbine/gold-data-for-ml")
  selected_features = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "Speed", "status"]
  
  # IN CASE YOU'D LIKE TO JUMP DIRECTLY TO AutoML AFTER INGESTION/DATA-ENGINEERING DEMO
  df = dataset.select(*selected_features).dropna()
  df.write \
    .format("delta") \
    .save(f"{path}/turbine/gold/automl_data")
  
  spark.sql("DROP TABLE IF EXISTS turbine_gold_for_automl")
  spark.sql(f"""CREATE TABLE turbine_gold_for_automl
  LOCATION '{path}/turbine/gold/automl_data'
  """)
  
  # Create "Gold" table for ML purpose and demo
  spark.sql("DROP TABLE IF EXISTS turbine_gold_for_ml")
  spark.sql(f"""CREATE TABLE turbine_gold_for_ml
  LOCATION '/mnt/quentin-demo-resources/turbine/gold-data-for-ml'
  """)
