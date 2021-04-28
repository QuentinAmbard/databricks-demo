# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance: model training
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-flow.png" width="90%"/>
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC One the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Use ML and MLFlow to detect damaged turbine
# MAGIC 
# MAGIC Our data is now ready. We'll now train a model to detect damaged turbines.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for ourturbines? 
# MAGIC 
# MAGIC *Notice the much larger stdev in AN8, AN9 and AN10 for Damaged turbined.*

# COMMAND ----------

dataset = spark.read.load("/mnt/quentin-demo-resources/turbine/gold-data-for-ml")
display(dataset)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: Workflows with Pyspark.ML Pipeline

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

with mlflow.start_run():
  #the source table will automatically be logged to mlflow
  mlflow.spark.autolog()
  
  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 42)
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5]).build()

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
  
  mlflow.spark.log_model(pipelineTrained, "turbine_gbt", input_example={"AN3":-1.4746, "AN4":-1.8042, "AN5":-2.1093, "AN6":-5.1975, "AN7":-0.45691, "AN8":-7.0763, "AN9":-3.3133, "AN10":-0.0059799})
  mlflow.set_tag("model", "turbine_gbt") 
  
  #Add confusion matrix to the model:
  labels = pipelineTrained.stages[2].labels
  fig = plt.figure()
  sn.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
  plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  mlflow.log_figure(fig, "confusion_matrix.png") #Requires MLFlow 1.13 (%pip install mlflow==1.13.1)
  

# COMMAND ----------

# MAGIC %md ## Saving our model to MLFLow registry

# COMMAND ----------

# DBTITLE 1,Save our new model to the registry as a new version
#get the best model having the best metrics.AUROC from the registry
best_models = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED" and metrics.f1 > 0', order_by=['metrics.f1 DESC'], max_results=1)
model_registered = mlflow.register_model("runs:/" + best_models.iloc[0].run_id + "/turbine_gbt", "turbine_gbt")

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(name = "turbine_gbt", version = model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detecting damaged turbine in a production pipeline

# COMMAND ----------

# DBTITLE 1,Load the model from our registry
model_from_registry = mlflow.spark.load_model('models:/turbine_gbt/production')

# COMMAND ----------

# DBTITLE 1,Let's call our model and make our predictive maintenance
predictions = model_from_registry.transform(dataset).select("ID", "prediction")
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC %md 
# MAGIC ### We can now explore our prediction in a new dashboard
# MAGIC 
# MAGIC ![turbine-demo-dashboard](https://github.com/QuentinAmbard/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-dashboard2.png)
# MAGIC 
# MAGIC [Open SQL Analytics dashboard](https://e2-demo-west.cloud.databricks.com/sql/dashboards/92d8ccfa-10bb-411c-b410-274b64b25520-turbine-demo-predictions?o=2556758628403379)
