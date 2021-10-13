# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Synchronizing your Data Lake from a SQL Database
# MAGIC 
# MAGIC Delta Lake is an <a href="https://delta.io/" target="_blank">open-source</a> storage layer with Transactional capabilities and increased Performances. 
# MAGIC 
# MAGIC Delta lake is designed to support CDC workload by providing support for UPDATE / DELETE and MERGE operation.
# MAGIC 
# MAGIC In addition, Delta table can support CDC to capture internal changes and propagate the changes downstream.
# MAGIC 
# MAGIC #### CDC flow
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail-cdc-forecast/resources/images/cdc_data_flow_high_level.png" alt='Make all your data ready for BI and ML'/>                             

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

display(spark.read.format("csv").schema("name string, address string, email string, id long, operation string, operation_date timestamp").load('/mnt/quentin-demo-resources/retail/clients/raw_cdc'))

# COMMAND ----------

# DBTITLE 1,We need to keep the cdc, however csv isn't a efficient storage. Let's put that in a Delta table instead:
bronzeDF = spark.readStream \
                .format("cloudFiles") \
                .option("cloudFiles.format", "csv") \
                .option("cloudFiles.maxFilesPerTrigger", 1) \
                .schema("name string, address string, email string, id long, operation string, operation_date timestamp") \
                .load("/mnt/quentin-demo-resources/retail/clients/raw_cdc") 
                  
bronzeDF.withColumn("file_name", input_file_name()).writeStream \
        .trigger(processingTime='10 seconds') \
        .format("delta") \
        .table("clients_cdc")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM clients_cdc order by id asc

# COMMAND ----------

# DBTITLE 1,We can now create our client table using standard SQL command
# MAGIC %sql 
# MAGIC -- we can add NOT NULL in our ID field (or even more advanced constraint)
# MAGIC CREATE TABLE IF NOT EXISTS retail_client_silver (id BIGINT NOT NULL, name STRING, address STRING, email STRING, operation_date TIMESTAMP) USING delta TBLPROPERTIES (delta.enableChangeDataCapture = true);

# COMMAND ----------

# DBTITLE 1,And run our MERGE statement the upsert the CDC information in our final table
def merge_stream(df, i):
  df.createOrReplaceTempView("clients_cdc_microbatch")
  #First we need to dedup the incoming data based on ID (we can have multiple update of the same row in our incoming data)
  #Then we run the merge (upsert or delete). We could do it with a window and filter on rank() == 1 too
  df._jdf.sparkSession().sql("""MERGE INTO retail_client_silver target
                                USING
                                (select id, name, address, email, operation, operation_date from 
                                  (SELECT *, RANK() OVER (PARTITION BY id ORDER BY operation_date DESC) as rank from clients_cdc_microbatch) 
                                 where rank = 1
                                ) as source
                                ON source.id = target.id
                                WHEN MATCHED AND source.operation = 'DELETE' THEN DELETE
                                WHEN MATCHED AND source.operation != 'DELETE' THEN UPDATE SET *
                                WHEN NOT MATCHED AND source.operation != 'DELETE' THEN INSERT *""")
  
spark.readStream \
       .table("clients_cdc") \
     .writeStream \
       .foreachBatch(merge_stream) \
       .trigger(processingTime='10 seconds') \
     .start()

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from retail_client_silver order by id asc;

# COMMAND ----------

# DBTITLE 1,Let's UPDATE id=1 and DELETE the row with id=2
# MAGIC %sql 
# MAGIC insert into clients_cdc  values ("Quentin", "Paris 75020", "quentin.ambard@databricks.com", 1, "UPDATE", now(), null);
# MAGIC insert into clients_cdc  values (null, null, null, 2, "DELETE", now(), null);
# MAGIC select * from clients_cdc where id in (1, 2);

# COMMAND ----------

# DBTITLE 1,Wait a few seconds for the stream to catch the new entry in the CDC table and check the results in the main table
# MAGIC %sql 
# MAGIC select * from retail_client_silver order by id asc;

# COMMAND ----------

# DBTITLE 1,The table history contains all our different versions in the silver table
# MAGIC %sql 
# MAGIC DESCRIBE HISTORY retail_client_silver;
# MAGIC -- If needed, we can go back in time to select a specific version or timestamp
# MAGIC -- SELECT * FROM retail_client_silver TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- And restore a given version
# MAGIC -- RESTORE retail_client_silver TO TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- Or clone the table (zero copy)
# MAGIC -- CREATE TABLE retail_client_silver_clone [SHALLOW | DEEP] CLONE retail_client_silver VERSION AS OF 32

# COMMAND ----------

# MAGIC %md ### We can now start to explore our data using Databricks SQL Analytics and build dashboard directly using our Silver tables:
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail-cdc-forecast/resources/images/retail_dashboard.png" alt='Make all your data ready for BI and ML' width="600px"/>
# MAGIC 
# MAGIC https://e2-demo-west.cloud.databricks.com/sql/dashboards/8deaa54a-482e-455e-91fc-ae62e89decbf-sales-report?o=2556758628403379

# COMMAND ----------

# MAGIC %md
# MAGIC ### But what about capturing the changes from our Silver table and streaming these changes downstream?
# MAGIC 
# MAGIC Let's use Delta CDC on our Delta table directly!
