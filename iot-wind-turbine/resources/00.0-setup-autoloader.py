# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"])

# COMMAND ----------

# MAGIC %run ./00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

dbutils.fs.rm(path+"/turbine/incoming-data-json", True) 
spark.read.format("json").option("inferSchema", "true").load("/mnt/quentin-demo-resources/turbine/incoming-data-json").limit(10).repartition(1).write.format("json").mode("overwrite").save(path+"/turbine/incoming-data-json")
