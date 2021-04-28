// Databricks notebook source
// MAGIC %md
// MAGIC #Kinesis producer
// MAGIC 
// MAGIC Use this producer to create a stream of fake user in your website.
// MAGIC 
// MAGIC The producer needs the following library to be installed in the cluster:
// MAGIC 
// MAGIC - `software.amazon.awssdk:kinesis:2.9.5`
// MAGIC - `com.amazonaws:amazon-kinesis-producer:0.13.1`

// COMMAND ----------

import java.nio.ByteBuffer
import java.util.{Properties, UUID}

import com.amazonaws.auth.AWSStaticCredentialsProvider
import com.amazonaws.services.kinesis.producer.{KinesisProducer, KinesisProducerConfiguration}

import scala.collection.mutable
import scala.util.Random

val userCreationRate = 5

val useKinesis = true

object kinesis {
  val config = new KinesisProducerConfiguration()
  config.setRegion("us-west-2")
  import com.amazonaws.auth.BasicAWSCredentials

  val awsCreds = new BasicAWSCredentials(dbutils.secrets.get("demo-semea-do-not-delete","aws-kinesis-key"), dbutils.secrets.get("demo-semea-do-not-delete","aws-kinesis-secret"))
  config.setCredentialsProvider(new AWSStaticCredentialsProvider(awsCreds))
  config.setMaxConnections(1)
  config.setRequestTimeout(60000)
  val producer = new KinesisProducer(config)

}

def sendMessage(msg: String): Unit = {
   kinesis.producer.addUserRecord("quentin-kinesis-demo", r.nextInt(10) + "", ByteBuffer.wrap(msg.getBytes("UTF-8")))
}

case class User(id: String, startDate: Long, endDate: Long)

val r = Random
val users = new mutable.HashMap[String, User]()

for (i <- 1 to 1000000) {
  users.foreach(u => {
    val now = System.currentTimeMillis()
    if (u._2.endDate < now) {
      users.remove(u._1)
      println(s"User ${u._1} removed")
    } else {
      //10% chance to click on something
      if (r.nextInt(100) > 80) {
        val randomInt = r.nextInt(100)
        val platform = if(randomInt > 97) "tablet" else if(randomInt > 60) "mobile" else "web"
        val msg = s"""{"user_uuid":"${u._2.id}", "page_id": ${r.nextInt(100)}, "timestamp": ${System.currentTimeMillis() / 1000}, "platform": "$platform", "geo_location":"Paris", "traffic_source": 1}"""
        sendMessage(msg)
        println(s"User ${u._1} sent event $msg")
      }
    }
  })
  if(useKinesis){
    kinesis.producer.flush()
  }
  //Re-create new users
  (1 to userCreationRate).foreach(_ => {
    //Add new user
    val uuid = UUID.randomUUID().toString
    users.put(uuid, User(uuid, System.currentTimeMillis(), System.currentTimeMillis() + r.nextInt(30000)))
    //println(s"User ${uuid} created")
  })
  Thread.sleep(1000)
}
println("closed")
