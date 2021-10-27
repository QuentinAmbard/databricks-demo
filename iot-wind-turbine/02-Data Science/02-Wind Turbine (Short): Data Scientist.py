# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/aelhelouDB/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-flow-ds.png" width="90%"/>
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC Our dataset consists of aggregated stats from vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# MAGIC %run ../resources/00-setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 
# MAGIC 
# MAGIC _Plot as bar charts using `summary` as Keys and all sensor Values_

# COMMAND ----------

dataset = spark.read.table("turbine_gold_for_ml")
display(dataset.summary().where(col("summary").isin(["mean","stddev"])))

# COMMAND ----------

# DBTITLE 1,Visualize Feature Distributions
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

selected_features = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]

def hide_current_axis(*args, **kwds):
  plt.gca().set_visible(False)

pairplot_pd = dataset.select(['status'] + list(map(lambda c: f"`{c}`", selected_features))).toPandas()
g = sns.pairplot(pairplot_pd, hue='status', vars=selected_features, dropna=True, palette='Paired')
g.map_upper(hide_current_axis)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train Model and Track Experiments

# COMMAND ----------

import mlflow
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.mllib.evaluation import MulticlassMetrics

with mlflow.start_run():
  #the source table will automatically be logged to mlflow
  mlflow.spark.autolog()
  
  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 5)
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5,10,15,25,30]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="status", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  
  mlflow.log_metric("precision", metrics.precision(1.0))
  mlflow.log_metric("recall", metrics.recall(1.0))
  mlflow.log_metric("f1", metrics.fMeasure(1.0))
  
  mlflow.spark.log_model(pipelineTrained, "turbine_gbt")
  mlflow.set_tag("model", "turbine_gbt")

# COMMAND ----------

# MAGIC %md ## Save to the model registry
# MAGIC Get the model having the best metrics.AUROC from the registry

# COMMAND ----------

best_models = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.f1 > 0', order_by=['metrics.f1 DESC'], max_results=1)
model_uri = best_models.iloc[0].artifact_uri+"/turbine_gbt"
model_registered = mlflow.register_model(model_uri, f"turbine_failure_model_{dbName}")
# print(model_uri)

# COMMAND ----------

# DBTITLE 1,Flag version as staging/production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(name = f"turbine_failure_model_{dbName}", version = model_registered.version, stage = "Staging", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC %md 
# MAGIC ### We can now explore our prediction in a new dashboard
# MAGIC 
# MAGIC [Open SQL Analytics dashboard](https://e2-demo-west.cloud.databricks.com/sql/dashboards/92d8ccfa-10bb-411c-b410-274b64b25520-turbine-demo-predictions?o=2556758628403379)
