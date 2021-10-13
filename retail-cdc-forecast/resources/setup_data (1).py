# Databricks notebook source
#dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"])

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

import re
from pyspark.sql.functions import rand, input_file_name, from_json, col, lit
from pyspark.sql.types import *

import pyspark.sql.functions as F

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/demo".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("using path {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/demo/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))

# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"
if reset_all:
  dbutils.fs.rm("""{}/demo/retail/clients/silver/data""".format(path), True)
  spark.read.format("delta").load('/mnt/'+mount_name+'/retail/clients/delta').write.format("delta").mode("overwrite").save("""{}/demo/retail/clients/silver/data""".format(path))
  spark.sql("""drop table if exists {}.retail_client_bronze_cdc""".format(dbName))
  spark.sql("""drop table if exists {}.retail_client_silver""".format(dbName))
  spark.sql("""create table if not exists {}.retail_client_silver using delta location '{}/demo/retail/clients/silver/data' """.format(dbName, path))

# COMMAND ----------

# MAGIC %md ####Data setup properly
