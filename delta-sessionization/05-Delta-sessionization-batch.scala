// Databricks notebook source
// MAGIC %md
// MAGIC #Batch mode to replay all the history
// MAGIC 
// MAGIC ### What if we want to re-process the entire history, or support long shut-down in our job?
// MAGIC 
// MAGIC If we want to make our job more resilient, we need to make it work for a batch where we would process all the events from the last months at once.
// MAGIC 
// MAGIC In this case, the `Iterator[ClickEvent]` in the `updateState` function could contain all the events from the begining. The implementation must be able to detect session within the iterator and return N UserSession.
// MAGIC 
// MAGIC This mean that inside the `updateState` function, the iterator must be something like that:
// MAGIC 
// MAGIC ```
// MAGIC ...
// MAGIC val sessions = List()
// MAGIC val prevUserSession = state.getOption.getOrElse {
// MAGIC   UserSession(userUuid, "online", new Timestamp(6284160000000L), new Timestamp(6284160L))
// MAGIC }
// MAGIC while(iterator.hasNext()){
// MAGIC   next = iterator.next
// MAGIC   ...
// MAGIC   //within the iterator, if there is an activity gap we need to create 2 sessions 
// MAGIC   if (prevUserSession.end.getTime  + 45 * 1000 < next.getTime){
// MAGIC      sessions.add(prevUserSession)
// MAGIC      prevUserSession = createNewSession(next)
// MAGIC   }
// MAGIC   ...
// MAGIC }
// MAGIC ...
// MAGIC return sessions
// MAGIC 
// MAGIC ```

// COMMAND ----------

//TODO!
