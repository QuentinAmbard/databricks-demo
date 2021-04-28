# Databricks notebook source
# MAGIC  %md
# MAGIC # CDC With Delta Table
# MAGIC **With Delta Table, you can capture the changes and propagate them downstream (DBR 8.2+) **
# MAGIC 
# MAGIC That's what we need to do from the Silver table to the Gold table in this example:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail-cdc-forecast/resources/images/cdc_data_flow_high_level.png" alt='Make all your data ready for BI and ML'/>                             

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# DBTITLE 1,Let's enable CDC in our Delta table
# MAGIC %sql 
# MAGIC ALTER TABLE retail_client_silver SET TBLPROPERTIES (delta.enableChangeDataCapture = true)
# MAGIC 
# MAGIC -- Note: you can also turn it on globally with: spark.databricks.delta.properties.defaults.enableChangeDataCapture = true

# COMMAND ----------

# DBTITLE 1,Let's see all our changes, from delta version 0:
# MAGIC %sql
# MAGIC SELECT * FROM table_changes('retail_client_silver', 2) 

# COMMAND ----------

# DBTITLE 1,It also works using a range of version or date:
# MAGIC %sql 
# MAGIC SELECT * FROM table_changes('retail_client_silver', '2021-03-19 15:50:39', '2021-03-19 15:51:39')

# COMMAND ----------

# MAGIC %md ## Delta CDC gives back 4 cdc types in the "_change_type" column:
# MAGIC 
# MAGIC | CDC Type             | Description                                                               |
# MAGIC |----------------------|---------------------------------------------------------------------------|
# MAGIC | **update_preimage**  | Content of the row before an update                                       |
# MAGIC | **update_postimage** | Content of the row after the update (what you want to capture downstream) |
# MAGIC | **delete**           | Content of a row that has been deleted                                    |
# MAGIC | **insert**           | Content of a new row that has been inserted                               |
# MAGIC 
# MAGIC Therefore, 1 update will result in 2 rows in the cdc stream (one row with the previous values, one with the new values)

# COMMAND ----------

# MAGIC %md ###Let's run some DELETE and UPDATE in our table to see the changes:

# COMMAND ----------

# MAGIC %sql 
# MAGIC DELETE FROM retail_client_silver WHERE id = 1;
# MAGIC UPDATE retail_client_silver SET name='Marwa' WHERE id = 13;

# COMMAND ----------

# DBTITLE 1,To get the UPDATE and DELETE CDC, we only select the last 2 version from our Delta Table:
last_version = str(DeltaTable.forName(spark, "retail_client_silver").history(1).head()["version"])
print("our Delta table last version is {}, let's select the last changes to see our DELETE and UPDATE operations (last 2 versions):".format(last_version))

changes = spark.sql("SELECT * FROM table_changes('retail_client_silver', {}-1)".format(last_version))
display(changes)

# COMMAND ----------

# MAGIC %md ###Synchronizing our downstream GOLD table based on the CDC from the Silver Delta Table
# MAGIC **Streaming** operations with CDC will only be supported with DBR 8.1. For now, we have to work with batch only.

# COMMAND ----------

# DBTITLE 1,Let's create or final GOLD table: retail_client_gold
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS retail_client_gold (id BIGINT NOT NULL, name STRING, address STRING, email STRING, operation_date TIMESTAMP) USING delta;

# COMMAND ----------

# DBTITLE 1,Reading the CDC using the Python API
cdc_values = spark.read.format("delta") \
                       .option("readChangeData", "true") \
                       .option("startingVersion", int(last_version) -1) \
                       .table("retail_client_silver")
#.option("endingVersion", int(last_version) -1) \

cdc_values.createOrReplaceTempView("cdc_values")
display(cdc_values)

# COMMAND ----------

# DBTITLE 1,MERGE the cdc to the gold table. We need to exclude "update_preimage" 
# MAGIC %scala
# MAGIC // Function to upsert `microBatchOutputDF` into Delta table using MERGE
# MAGIC def upsertToDelta(data: DataFrame, batchId: Long) {
# MAGIC   //First we need to deduplicate based on the id and take the most recent update
# MAGIC   val windowSpec = Window.partitionBy("id").orderBy($"_commit_version".desc)
# MAGIC   //Select only the first value 
# MAGIC   //getting the latest change is still needed if the cdc contains multiple time the same id. We can rank over the id and get the most recent _commit_version
# MAGIC   var data_deduplicated = data.withColumn("rank", rank().over(windowSpec)).where("rank = 1 and _change_type!='update_preimage'").drop("_commit_version", "rank")
# MAGIC 
# MAGIC   //Add some data cleaning for the gold layer to remove quotes from the address
# MAGIC   data_deduplicated = data_deduplicated.withColumn("address", regexp_replace($"address", "\"", ""))
# MAGIC   
# MAGIC   //run the merge in the gold table directly
# MAGIC   DeltaTable.forName("retail_client_gold").as("target")
# MAGIC       .merge(data_deduplicated.as("source"), "source.id = target.id")
# MAGIC       .whenMatched("source._change_type = 'delete'").delete()
# MAGIC       .whenMatched("source._change_type != 'delete'").updateAll()
# MAGIC       .whenNotMatched("source._change_type != 'delete'").insertAll()
# MAGIC       .execute()
# MAGIC }
# MAGIC 
# MAGIC spark.readStream
# MAGIC        .option("readChangeData", "true")
# MAGIC        .option("startingVersion", 1)
# MAGIC        .table("retail_client_silver")
# MAGIC       .writeStream
# MAGIC         .foreachBatch(upsertToDelta _)
# MAGIC       .start()

# COMMAND ----------

# MAGIC %sql select * from retail_client_gold

# COMMAND ----------

# MAGIC %md
# MAGIC ### Our gold table is now ready!
# MAGIC We can improve our dashboards and start creating proper ML Models based on this information, for example:
# MAGIC 
# MAGIC - Sales forecast for stock prediction
# MAGIC - Customer segmentation to understand customer database and apply specific targeting
