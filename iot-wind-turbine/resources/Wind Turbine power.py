# Databricks notebook source
display(spark.read.format("text").load("/mnt/quentin-demo-resources/turbine/power/raw"))

# COMMAND ----------

#Load stream from our files
spark.readStream.format("cloudFiles") \
                .option("cloudFiles.format", "json") \
                .option("cloudFiles.maxFilesPerTrigger", 1) \
                .schema("turbine_id bigint, date timestamp, power float, wind_speed float, theoretical_power_curve float, wind_direction float") \
                .load("/Users/quentin.ambard@databricks.com/turbine/power/raw") \
     .writeStream.format("delta") \
        .option("checkpointLocation", "/Users/quentin.ambard@databricks.com/turbine/power/bronze/checkpoint") \
        .option("path", "/Users/quentin.ambard@databricks.com/turbine/power/bronze/data") \
        .trigger(processingTime = "10 seconds") \
        .start()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists quentin.turbine_power_bronze;
# MAGIC -- Add the table in our data catalog
# MAGIC create table if not exists quentin.turbine_power_bronze
# MAGIC   using delta
# MAGIC   location '/Users/quentin.ambard@databricks.com/turbine/power/bronze/data'
# MAGIC   TBLPROPERTIES ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);
# MAGIC 
# MAGIC -- Select data
# MAGIC select to_date(date) date, sum(power) as power from quentin.turbine_power_bronze group by date;