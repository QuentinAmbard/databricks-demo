# Databricks notebook source
# MAGIC %run ./00-setup $reset_all=$reset_all

# COMMAND ----------

dbutils.fs.rm(path+"/turbine/incoming-data-json", True)
spark.read.format("json").option("inferSchema", "true").load("/mnt/quentin-demo-resources/turbine/incoming-data-json").limit(10).repartition(1).write.format("json").mode("overwrite").save(path+"/turbine/incoming-data-json")