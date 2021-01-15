# Databricks notebook source
# MAGIC %fs ls /datasets

# COMMAND ----------

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

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/demo".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("using path {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/turbine/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))


# COMMAND ----------

tables = ["turbine_bronze", "turbine_silver", "turbine_gold", "turbine_power", "turbine_schema_evolution"]
reset_all = dbutils.widgets.get("reset_all") == "true" or any([not spark.catalog._jcatalog.tableExists(table) for table in ["turbine_power"]])
if reset_all:
  print("resetting data")
  for table in tables:
    spark.sql("""drop table if exists {}.{}""".format(dbName, table))
    
  #spark.sql("""drop database if exists {} CASCADE""".format(dbName))
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
else:
  print("loaded without data reset")

  
# Define the default checkpoint location to avoid managing that per stream and making it easier. In production it can be better to set the location at a stream level.
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/turbine/_checkpoint")

#Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")