// Databricks notebook source
// MAGIC %md
// MAGIC # ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)  4/ Run UPDATE and DELETE in our Delta Lake 
// MAGIC 
// MAGIC ### DELETING data for GDPR reason
// MAGIC For compliance reason, we're asked to delete all the data we have from our customer whith `UUID = "8c662172-22d2-4bf8-9511-fea8a04f4cea"`
// MAGIC 
// MAGIC While this is impossible with a Parquet table, it's very simple with DELTA!
// MAGIC 
// MAGIC _Note: Ideally, we would run this operation from another cluster to leave all resources to the streaming cluster_

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Let's delete this user from the SILVER table:

// COMMAND ----------

// MAGIC %sql 
// MAGIC delete from quentin.events_silver WHERE user_uuid = '1048e2a0-3ac7-4267-8b79-efee689d69b3'

// COMMAND ----------

// MAGIC %md 
// MAGIC ### And delete from the BRONZE table:

// COMMAND ----------

// MAGIC %sql 
// MAGIC delete from quentin.events_bronze WHERE sequenceNumber IN (
// MAGIC     select sequenceNumber from (
// MAGIC       select sequenceNumber, from_json(cast(data as String), 'user_uuid String, page_id Int, timestamp Timestamp, platform String, geo_location String, traffic_source Int') as data from quentin.events_bronze) 
// MAGIC     where data.user_uuid = '34d63070-01b9-4304-8bb4-192a67436dd5')

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Ouups? Did we just delete the wrong ID? Let's use Delta Time Travel to go back in time
// MAGIC 
// MAGIC Need to undo your delete or update? You can easily get back your data using Delta Time Travel.
// MAGIC 
// MAGIC _Note: If you are 100% sure of you operation and want to clean your history, just run `VACUUM quentin.events_silver RETAIN 24 hours`_

// COMMAND ----------

// MAGIC %sql 
// MAGIC describe history quentin.events_silver

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from quentin.events_silver version as of 14

// COMMAND ----------

// MAGIC %md
// MAGIC ### Update with MERGE INTO
// MAGIC SQL `MERGE INTO`can also be used with DELTA to update existing rows (based on an identifier) or insert the row if it doesn't exist yet !

// COMMAND ----------

// MAGIC %md 
// MAGIC ### What about our streams? Did we break something during our update?
// MAGIC 
// MAGIC Because DELTA provides **ACID TRANSACTIONS**, we can safely run our DELETE while our streams are running as a background task!
// MAGIC 
// MAGIC  **[Advanced sessionization with batch mode](https://demo.cloud.databricks.com/#notebook/4422451)**
// MAGIC 
// MAGIC **[Go Back](https://demo.cloud.databricks.com/#notebook/4438519)**
