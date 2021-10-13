# Databricks notebook source
# MAGIC %md # Auto-loader schema inference & error handling demo

# COMMAND ----------

# DBTITLE 1,Let's prepare our data first
# MAGIC %run ./resources/00.0-setup-autoloader $reset_all_data=$reset_all_data

# COMMAND ----------

# DBTITLE 1,Let's explore what is being delivered by our wind turbines stream: (json)
# MAGIC %sql 
# MAGIC select * from text.`/mnt/quentin-demo-resources/turbine/incoming-data-json`

# COMMAND ----------

# MAGIC %md ### Schema inference

# COMMAND ----------

# DBTITLE 1,Previously, autoloader wouldn't infer schema on parquet/json/... 
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "false") #false is default
bronzeDF = spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", "json") \
                .load("/mnt/quentin-demo-resources/turbine/incoming-data-json") 
display(bronzeDF)

# COMMAND ----------

# DBTITLE 1,Autoloader can now infer the schema automatically (from any format) 
#Note: schema is infered from a sample of files, making inference faster
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")
bronzeDF = spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", "json") \
                .load("/mnt/quentin-demo-resources/turbine/incoming-data-json") 
display(bronzeDF)

# COMMAND ----------

# MAGIC %md ### Schema evolution

# COMMAND ----------

# DBTITLE 1,Schema evolution is now supported by restarting the stream
stream = (spark.readStream 
                .format("cloudFiles") 
                .option("cloudFiles.format", "json") 
                #will fail and restart the stream when new columns appear 
                .option("failOnUnknownFields", "true") 
                # enforce schema on part of the field (ex: for date or specific FLOAT types)  
                .option("cloudFiles.schemaHints", "TIMESTAMP TIMESTAMP, infos STRUCT<wind_speed:FLOAT, wind_direction:STRING>") 
                # will collect columns where the data types can change across rows 
                .option("unparsedDataColumn", "_incorrect_data")
                .load(path+"/turbine/incoming-data-json"))
    
display(stream)

# COMMAND ----------

# DBTITLE 1,Adding a row with an extra column ("new_column": "test", infos": {"wind_speed": 0.123, "wind_direction": "west"})
new_row = spark.read.json(sc.parallelize(['{"AN3":-1.4746,"AN4":-1.8042,"AN5":-2.1093,"AN6":-5.1975,"AN7":-0.45691,"AN8":-7.0763,"AN9":-3.3133,"AN10":-0.0059799,"SPEED":4.8805,"ID":347.0,"TIMESTAMP":"2020-06-07T19:18:46.000Z", "new_column": "test", "infos": {"wind_speed": 0.123, "wind_direction": "west"}}']))
new_row.write.format("json").mode("append").save(path+"/turbine/incoming-data-json")

# COMMAND ----------

# DBTITLE 1,Let's add an incorrect field ("infos.wind_speed" and "AN3" as string instead of float)
incorrect_data = spark.read.json(sc.parallelize(['{"AN3":"-1.4746","AN4":-1.8042,"AN5":-2.1093,"AN6":-5.1975,"AN7":-0.45691,"AN8":-7.0763,"AN9":-3.3133,"AN10":-0.0059799,"SPEED":4.8805,"ID":347.0,"TIMESTAMP":"2020-06-07T19:18:46.000Z", "infos": {"wind_speed": "fast_wind", "wind_direction": "south"}}']))
incorrect_data.write.format("json").mode("append").save(path+"/turbine/incoming-data-json")

# COMMAND ----------

# MAGIC %md ### Writting the data to a Delta table, supporting schema changes & evolution:

# COMMAND ----------

def start_stream():
  return (spark.readStream 
              .format("cloudFiles") 
              .option("cloudFiles.format", "json") 
              #will fail and restart the stream when new columns appear 
              .option("failOnUnknownFields", "true") 
              # enforce schema on part of the field (ex: for date or specific FLOAT types)  
              .option("cloudFiles.schemaHints", "TIMESTAMP TIMESTAMP, infos STRUCT<wind_speed:FLOAT, wind_direction:STRING>") 
              # will collect columns where the data types can change across rows 
              .option("unparsedDataColumn", "_incorrect_data")
              .load(path+"/turbine/incoming-data-json"))

def start_stream_restart_on_schema_evolution():
  while True:
    try:
      bronzeDF = start_stream()
      q = bronzeDF.writeStream \
          .format("delta") \
          .option("mergeSchema", "true") \
          .table("turbine_schema_evolution")
      q.awaitTermination()
      return q
    except BaseException as e:
      if not ('UnknownFieldException' in str(e.stackTrace)):
        raise e
        
start_stream_restart_on_schema_evolution()

# COMMAND ----------

# MAGIC %sql select * from turbine_schema_evolution
