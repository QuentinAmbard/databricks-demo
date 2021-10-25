# Databricks notebook source
# MAGIC %md #Table for SQL Analytics loader
# MAGIC ##Run this notebook to mount the final table and directly run queries on top of them
# MAGIC The tables are in a separate database from the one in the data ingestion notebook to prevent conflict, so that we can run SELECT queries with SQL Analytics  

# COMMAND ----------

# MAGIC %md ### Please don't delete/edit these table, just create them to access them on SQL Analytics, don't edit them on the demo notebooks 

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

# MAGIC %sql
# MAGIC create database if not exists demo_turbine;
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_bronze` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/bronze/data';
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_silver` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/silver/data';
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_gold`   USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/gold/data' ;
# MAGIC 
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_power_prediction` USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/power/prediction/data';
# MAGIC CREATE TABLE if not exists `demo_turbine`.`turbine_power_bronze`     USING delta LOCATION 'dbfs:/mnt/quentin-demo-resources/turbine/power/bronze/data';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Select data
# MAGIC select to_date(date) date, sum(power) as power from quentin.turbine_power_bronze group by date;
