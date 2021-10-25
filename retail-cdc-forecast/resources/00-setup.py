# Databricks notebook source
#dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"])

# COMMAND ----------

from delta.tables import *
import pandas as pd
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)
from pyspark.sql.functions import to_date, col
import tempfile
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType, input_file_name
import re

# COMMAND ----------

aws_bucket_name = "quentin-demo-resources"
mount_name = "quentin-demo-resources"

#dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)
try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  print("bucket isn't mounted, mount the demo bucket under %s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)

# COMMAND ----------

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/demo".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("using path {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/global_demo/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))

# COMMAND ----------

# Define the default checkpoint location to avoid managing that per stream and making it easier. In production it can be better to set the location at a stream level.
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/retail/_checkpoint")

#Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"
if reset_all:
  #dbutils.fs.rm("""{}/demo/retail/clients/silver/data""".format(path), True)
  #spark.read.format("delta").load('/mnt/'+mount_name+'/retail/clients/delta').write.format("delta").mode("overwrite").save("""{}/demo/retail/clients/silver/data""".format(path))
  spark.sql("""drop table if exists {}.retail_client_silver""".format(dbName))
  spark.sql("""drop table if exists {}.clients_cdc""".format(dbName))

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import org.apache.spark.sql.expressions.Window
# MAGIC import io.delta.tables._
# MAGIC import org.apache.spark.sql.DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC Data successfully initialized

# COMMAND ----------

#csv = spark.read.format("csv").schema("name string, address string, email string, id long, operation string, operation_date timestamp").load('/mnt/quentin-demo-resources/retail/clients/raw_cdc')
#import pyspark.sql.functions as F
#csv.withColumn("operation_date", F.from_unixtime(F.round(F.rand()*1000000+1610017617))).repartition(100).write.format("csv").mode("overwrite").save("/mnt/quentin-demo-resources/retail/clients/raw_cdc")

# COMMAND ----------

#spark.read.format("csv").schema("name string, address string, email string, id long, operation string, operation_date timestamp").load('/mnt/quentin-demo-resources/retail/clients/raw_cdc').orderBy("id").repartition(100).write.format("csv").mode("overwrite").save("/mnt/quentin-demo-resources/retail/clients/raw_cdc")
