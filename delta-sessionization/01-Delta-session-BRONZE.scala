// Databricks notebook source
// MAGIC %md-sandbox
// MAGIC 
// MAGIC #Detecting Building a sessionization stream with Delta Lake
// MAGIC ### What's sessionization?
// MAGIC <div style="float:right" ><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/sessionization/session_diagram.png" style="height: 200px; margin:0px 0px 0px 10px"/></div>
// MAGIC 
// MAGIC Sessionization is the process of finding time-bounded user session from a flow of event, grouping all events happening around the same time (ex: number of clicks, pages most view etc)
// MAGIC 
// MAGIC When there is a temporal gap greater than X minute, we decide to split the session in 2 distinct sessions
// MAGIC 
// MAGIC ### Why is that important?
// MAGIC 
// MAGIC Understanding sessions is critical for a lot of use cases:
// MAGIC 
// MAGIC - Detect cart abandonment in your online shot, and automatically trigger marketing actions as follow-up to increase your sales
// MAGIC - Build better attribution model for your affiliation, based on the user actions during each session 
// MAGIC - Understand user journey in your website, and provide better experience to increase your user retention
// MAGIC - ...
// MAGIC 
// MAGIC 
// MAGIC ### Sessionization with Spark & Delta
// MAGIC 
// MAGIC Sessionization can be done in many ways. SQL windowing is often used but quickly become too restricted for complex use-case. 
// MAGIC 
// MAGIC Instead, we'll be using the following Delta Architecture:
// MAGIC 
// MAGIC ![Delta Lake Tiny Logo](https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/sessionization/sessionization.png)
// MAGIC 
// MAGIC Being able to process and aggregate your sessions in a Batch and Streaming fashion can be a real challenge, especially when updates are required in your historical data!
// MAGIC 
// MAGIC Thankfully, Delta and Spark can simplify our job, using Spark Streaming function with a custom stateful operation (`flatMapGroupsWithState` operator), in a streaming and batch fashion.
// MAGIC 
// MAGIC Let's build our Session job to detect cart abandonment !

// COMMAND ----------

// MAGIC %md-sandbox
// MAGIC 
// MAGIC <img style="float:right; height: 220px; margin: 0px 30px 0px 30px" src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/sessionization/session_bronze.png">
// MAGIC # ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)  1/ Bronze table: store the stream in our Delta Lake
// MAGIC The first step is to consume data from our streaming engine (Kafka, Kinesis, Pulsar etc.) and save it in our Data Lake, in a Bronze table.
// MAGIC 
// MAGIC We won't be doing any transformation, the goal is to be able to re-process all the data and change/improve the downstream logic when needed

// COMMAND ----------

import spark.implicits._
import java.sql.Timestamp
import org.apache.spark.sql.catalyst.plans.logical.EventTimeTimeout
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.GroupState
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.streaming.Trigger
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
spark.conf.set("spark.sql.shuffle.partitions", "24")
spark.conf.set("spark.default.parallelism", "24")

// COMMAND ----------

//Clean all folders for a fresh start, recommended.
val levels = List("bronze", "silver", "gold")
List("/quentin/demo/session/", "/quentin/demo/session/checkpoint_").foreach(p => {
  levels.foreach(l => {
    println("deleting"+p+l)
    dbutils.fs.rm(p+l, true)
  })
 })
levels.foreach(l => spark.sql(s"drop table if exists quentin.$l"))

// COMMAND ----------

// == Managed Delta on Databricks: automatically compact small files during write:
spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", "true")

val stream = spark
  .readStream
     // === Configurations for Kinesis streams ===
    .format("kinesis").option("streamName", "quentin-kinesis-demo").option("region", "us-west-2").option("initialPosition", "LATEST")
    .option("awsAccessKey", dbutils.secrets.get("demo-semea-do-not-delete","quentin-demo-kinesis-aws-access-key"))
    .option("awsSecretKey", dbutils.secrets.get("demo-semea-do-not-delete","quentin-demo-kinesis-aws-secret-key"))
  .load()
  .coaelesce(4)
  .withCm(wxwxxxx)
  .writeStream
     // === Write to the delta table ===
    .format("delta")
    .trigger(Trigger.ProcessingTime("20 seconds"))
    .option("checkpointLocation", "/quentin/demo/session/checkpoint_bronze")
    .outputMode("append")
    .start("/quentin/demo/session/bronze")

// COMMAND ----------

// MAGIC %fs ls /quentin/demo/session/bronze/

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE TABLE IF NOT EXISTS quentin.events_bronze 
// MAGIC   USING delta 
// MAGIC   LOCATION "/quentin/demo/session/bronze"

// COMMAND ----------

// MAGIC %md
// MAGIC ##### It's now easy to run queries in our events_bronze table, however all our data is stored as JSON in a unique binary binary field, which isn't ideal...

// COMMAND ----------

// MAGIC %sql
// MAGIC select cast(data as String) as data, partitionKey, stream, shardId, sequenceNumber, approximateArrivalTimestamp from quentin.events_bronze;

// COMMAND ----------

// MAGIC %md
// MAGIC ## Interactive queries in our message content
// MAGIC We can always use spark spark to run queries on the raw data...

// COMMAND ----------

// MAGIC %sql
// MAGIC select data.* from (select from_json(cast(data as String), 'user_uuid String, page_id Int, timestamp Timestamp, platform String, geo_location String, traffic_source Int') as data from quentin.events_bronze);

// COMMAND ----------

// MAGIC %md 
// MAGIC ## But converting JSON from binary is slow and expensive, and what if our json changes over time ?
// MAGIC While we can explore the dataset using spark json manipulation, this isn't ideal. For example is the json in our message changes after a few month, our request will fail.
// MAGIC 
// MAGIC Futhermore, performances won't be great at scale: because all our data is stored as a unique binary column, we can't leverage data skipping and a columnar format
// MAGIC 
// MAGIC That's why we need another table:  **[A Silver Table!](https://demo.cloud.databricks.com/#notebook/4439040)**
