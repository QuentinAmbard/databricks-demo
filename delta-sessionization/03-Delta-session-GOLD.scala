// Databricks notebook source
// MAGIC %md-sandbox
// MAGIC # ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png)  3/ GOLD table: extract the sessions
// MAGIC <div style="float:right" ><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/sessionization/session_diagram.png" style="height: 280px; margin:0px 0px 0px 10px"/></div>
// MAGIC 
// MAGIC ### Why is this a challenge?
// MAGIC Because we don't have any event to flag the user disconnection, detecting the end of the session is hard. After 10 minutes without any events, we want to be notified that the session has ended.
// MAGIC However, spark will only react on event, not the absence of event.
// MAGIC 
// MAGIC Thanksfully, Spark Structured Streaming has the concept of timeout. 
// MAGIC 
// MAGIC **We can set a 10 minutes timeout in the state engine** and be notified 10 minutes later in order to close the session

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
spark.conf.set("spark.default.parallelism", "24")
//set the number of partitions to x2 our cluster cores 
spark.conf.set("spark.sql.shuffle.partitions", "24")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### To make things simple, we'll be using Scala case class to represent our Events and our Sesssions

// COMMAND ----------

//Event (from the silver table)
case class ClickEvent(user_uuid: String, page_id: Long, timestamp: Timestamp, platform: String, geo_location: String, traffic_source: Int)

//Session (from the gold table)
case class UserSession(userUuid: String, var status: String = "online", var start: Timestamp, var end: Timestamp, var clickCount: Integer = 0)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Implementing the aggregation function to update our Session
// MAGIC 
// MAGIC In this simple example, we'll just be counting the number of click in the session.
// MAGIC 
// MAGIC The function `updateState` will be called for each user with a list of events for this user.

// COMMAND ----------

def updateState(userUuid: String, events: Iterator[ClickEvent], state: GroupState[UserSession]): Iterator[UserSession] = {
  val prevUserSession = state.getOption.getOrElse {
    UserSession(userUuid, "online", new Timestamp(6284160000000L), new Timestamp(6284160L))
  }

  //Just an update, not a timeout
  if (!state.hasTimedOut) {
    events.foreach { event =>
      updateUserSessionWithEvent(prevUserSession, event)
    }
    //Renew the session timeout by 45 seconds
    state.setTimeoutTimestamp(prevUserSession.end.getTime + 45 * 1000)
    state.update(prevUserSession)
  } else {
    //Timeout, we flag the session as offline
    if (prevUserSession.status == "online") {
      //Update the stet as offline
      prevUserSession.status = "offline"
      state.update(prevUserSession)
    } else {
      //Session has been flagged as offline during the last run, we can safely discard it
      state.remove()
    }
  }
   return Iterator(prevUserSession)
}

def updateUserSessionWithEvent(state: UserSession, input: ClickEvent): UserSession = {
  state.status = "online"
  state.clickCount += 1
  //Update then begining and end of our session
  if (input.timestamp.after(state.end)) {
    state.end = input.timestamp
  }
  if (input.timestamp.before(state.start)) {
    state.start = input.timestamp
  }
  //return the updated state
  state
}

// COMMAND ----------

val stream = spark
.readStream
    .format("delta")
    .option("ignoreChanges", "true")
    .table("quentin.events_silver")  
  .as[ClickEvent]
  .withWatermark("timestamp", "45 seconds")
  .groupByKey(_.user_uuid)
  .flatMapGroupsWithState(OutputMode.Append(), EventTimeTimeout)(updateState)

display(stream)

// COMMAND ----------

// MAGIC %md
// MAGIC ### We now have our sessions stream running!
// MAGIC 
// MAGIC We can set the output of this streaming job to a SQL database or another queuing system.
// MAGIC 
// MAGIC We'll be able to automatically detect cart abandonments in our website and send an email to our customers, our maybe just give them a call asking if they need some help! 
// MAGIC 
// MAGIC But what if we have to delete the data for some of our customers ? **[DELETE and UPDATE with delta](https://demo.cloud.databricks.com/#notebook/4440322)**
// MAGIC 
// MAGIC 
// MAGIC **[Go Back](https://demo.cloud.databricks.com/#notebook/4439040)**
