// Databricks notebook source
// MAGIC %md-sandbox
// MAGIC 
// MAGIC # ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)  2/ SILVER table: store the content of our events in a structured table
// MAGIC <img style="float:right; height: 250px; margin: 0px 30px 0px 30px" src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/sessionization/session_silver.png">
// MAGIC 
// MAGIC We can create a new silver table containing all our data.
// MAGIC 
// MAGIC This will allow to store all our data in a proper table, with the content of the json stored in a columnar format. 
// MAGIC 
// MAGIC Should our message content change, we'll be able to adapt the transformation of this job to always allow SQL queries over this SILVER table.
// MAGIC 
// MAGIC If we realized our logic was flawed from the begining, it'll also be easy to start a new cluster to re-process the entire table with a better transformation!

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
//Event (from the silver table)
case class ClickEvent(user_uuid: String, page_id: Long, timestamp: Timestamp, platform: String, geo_location: String, traffic_source: Int)

// COMMAND ----------

// == Managed Delta on Databricks: automatically compact small files during write:
spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", "true")
val schema = ScalaReflection.schemaFor[ClickEvent].dataType.asInstanceOf[StructType]

spark
  .readStream
    .format("delta")
    .option("ignoreChanges", "true")
    .load("/quentin/demo/session/bronze")
   // === Our transformation, easy to adapt if our logic changes ===
  .withColumn("data", $"data".cast(StringType))
  .select(from_json($"data", schema).as("json")).select("json.*")
  .writeStream
     // === Write to the delta table ===
    .format("delta")
    .trigger(Trigger.ProcessingTime("20 seconds"))
    .option("checkpointLocation", "/quentin/demo/session/checkpoint_silver").outputMode("append").start("/quentin/demo/session/silver")

// COMMAND ----------

// MAGIC %sql
// MAGIC CREATE TABLE IF NOT EXISTS quentin.events_silver
// MAGIC   USING delta 
// MAGIC   LOCATION "/quentin/demo/session/silver"

// COMMAND ----------

// MAGIC %md
// MAGIC ##### It's now easy and fast to run interactive queries in our silver table:

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from quentin.events_silver

// COMMAND ----------

// MAGIC %md
// MAGIC #### Let's display a real-time view of our traffic using our stream, grouped by platform, for the last minute

// COMMAND ----------

spark.readStream.format("delta").option("ignoreChanges", "true").load("/quentin/demo/session/silver").createOrReplaceTempView("event_silver_stream")


// COMMAND ----------

// MAGIC %sql
// MAGIC select w.*, c, platform from (select window(timestamp, "10 seconds") as w, count(*) c, platform from event_silver_stream where CAST(timestamp as INT) > CAST(current_timestamp() as INT)-60 group by w, platform )

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Let's find our TOP 10 more active users, updated in real time:

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) as count, user_uuid from event_silver_stream group by user_uuid order by count desc limit 10;

// COMMAND ----------

// MAGIC %md
// MAGIC ### We now have our silver table ready to be used!
// MAGIC 
// MAGIC Let's compute our sessions based on this table with  **[a Gold Table](https://demo.cloud.databricks.com/#notebook/4438519)**
// MAGIC 
// MAGIC 
// MAGIC **[Go Back](https://demo.cloud.databricks.com/#notebook/4128443)**
