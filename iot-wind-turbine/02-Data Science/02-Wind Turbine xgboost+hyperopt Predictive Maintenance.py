# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance (using xgboost and hyperopt libraries)
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/aelhelouDB/databricks-demo/raw/main/iot-wind-turbine/resources/images/turbine-demo-flow-ds.png" width="90%" />
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *locations of the sensors*
# MAGIC </div>
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use **XGBOOST** a famous Gradient Boosted Tree Classification library to predict which set of vibrations could be indicative of a failure.
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
# MAGIC ## Read dataset
# MAGIC Reading as Spark Dataframe

# COMMAND ----------

dataset_raw = spark.read.table("turbine_gold_for_ml")

# COMMAND ----------

sensor_features = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE"]
output = "status"

# COMMAND ----------

# DBTITLE 1,View summary stats
display(dataset_raw[sensor_features].summary())

# COMMAND ----------

# DBTITLE 1,Check Class Imbalance
dataset_raw.select(output).groupBy(output).count().display()

# COMMAND ----------

# DBTITLE 1,Encode output as binary (for xgboost)
from pyspark.sql.functions import col, lit

dataset_cleanDF = dataset_raw.dropDuplicates(subset=["TIMESTAMP"]) \
                         .withColumn("Damaged", col("status") == lit("damaged")) \
                         .drop(output,"ID", "TIMESTAMP")

output = ["Damaged"]
labels = ["Healthy", "Damaged"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single Node quick EDA
# MAGIC Given small size of dataset gather into driver node as Pandas DataFrame and run a quick Exploratory Data Analysis

# COMMAND ----------

df_dataset = dataset_cleanDF.toPandas()

# COMMAND ----------

# DBTITLE 1,Import libraries of choice
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Visualize Correlations
sns.heatmap(df_dataset.corr())

# COMMAND ----------

# DBTITLE 1,Count Missing Data
df_dataset.isna().sum().plot(kind='bar')

# COMMAND ----------

# DBTITLE 1,Drop cols with missing rows
df_dataset_clean = df_dataset.dropna(axis=1, how='any')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: using xgboost, hyperopt and mlflow

# COMMAND ----------

# DBTITLE 1,Cross-Validation: Hold-Out
seed = 123 # For reproducibility
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_dataset_clean, random_state=seed)
X_train = train.drop(output, axis=1)
X_test = test.drop(output, axis=1)
y_train = train[output].astype('int')
y_test = test[output].astype('int')

# COMMAND ----------

# DBTITLE 1,Define training function with loss function to minimize
from math import exp
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import xgboost as xgb

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from mlflow.models.signature import infer_signature

model_name = f"turbine_failure_model_xgboost_{dbName}"

def f1_eval(y_pred, d_test):
  # Custom F1 wrapper for XGBOOST
  y_true = d_test.get_label()
  f1 = f1_score(y_true, np.round(y_pred))
  return f1
  
def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    
    # Train
    booster = xgb.train(params=params, dtrain=train, num_boost_round=20,\
                        evals=[(test, "test")], early_stopping_rounds=10)
    # Evaluate on test set
    predictions_test = booster.predict(test)
    
    # Calculate AUC, F1 & log values
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)
    f1_score_ = f1_eval(predictions_test, test)
    mlflow.log_metric('F1', f1_score_)

    # Log model signature (for traceability)
    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Add tag for searchability
    mlflow.set_tag("model", model_name) 
  
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# COMMAND ----------

# DBTITLE 1,Define search space and run trials in parallel on Spark cluster
search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 10, 30, 1)),
  'learning_rate': hp.loguniform('learning_rate', -5, -0.5), #between exp(-5)=0.0067 and exp(-0.5)=0.60
  'reg_alpha': hp.loguniform('reg_alpha', -5, -0.5),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -0.5),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': seed, # Set a seed for deterministic training
}

# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=4)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='xgboost_turbine_experiments'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=16,
    trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# DBTITLE 1,Create table/SQL view for ad-hoc experiment analysis
df_client = spark.read.format("mlflow-experiment").load()
df_client.createOrReplaceTempView("vw_client")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLOps: Query all runs
# MAGIC *can be done in a separate notebook by MLOps engineers*

# COMMAND ----------

# DBTITLE 1,Using SparkSQL
df_model_selector = spark.sql("""SELECT experiment_id, run_id, metrics.auc as AUC, metrics.F1 as F1, artifact_uri 

    FROM vw_client 
    
    WHERE status='FINISHED'
    ORDER BY metrics.f1 desc

  """)
display(df_model_selector)

# COMMAND ----------

# DBTITLE 1,Using MLFlow client API
best_models = mlflow.search_runs(
  filter_string=f'tags.model="{model_name}" and attributes.status = "FINISHED" and metrics.F1 > 0',
  order_by=['metrics.F1 DESC'], max_results=1
)

model_uri = best_models.iloc[0].artifact_uri
print(f'AUC of Best Run: {best_models["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLOps: Select best model and push to Registry

# COMMAND ----------

# DBTITLE 1,Instantiate MLFlow client for API calls
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# DBTITLE 1,Save model to the registry as a new version
model_registered = mlflow.register_model(best_models.iloc[0].artifact_uri+"/{model_name}", model_name)
print("registering model version "+model_registered.version)

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(
  name = model_name,
  version = model_registered.version,
  stage = "Production",
  archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference: Detecting damaged turbine in a production pipeline

# COMMAND ----------

# DBTITLE 1,Pull latest "Production" model
latest_prod_model_detail = client.get_latest_versions(model_name, stages=['Production'])[0]
print(latest_prod_model_detail)

# COMMAND ----------

# DBTITLE 1,Create a python User-Defined-Function (for scaling-out inference over large batch of data)
batch_inference_udf = mlflow.pyfunc.spark_udf(spark, latest_prod_model_detail.source)

# COMMAND ----------

# DBTITLE 1,Grab model inputs name
from pyspark.sql.functions import struct

model_input_names = mlflow.pyfunc.load_model(latest_prod_model_detail.source).metadata.signature.inputs.input_names()
udf_inputs = struct(*(model_input_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load new data
# MAGIC Read in new data from a specified *(delta)* location (presumably an Azure/ADLS location or some other datasource)

# COMMAND ----------

new_data_path = "/mnt/quentin-demo-resources/turbine/gold-data-for-ml"
df_new_data = spark.read \
.format("delta") \
.load(new_data_path) \
.drop("status")

# COMMAND ----------

# DBTITLE 1,Infer/Score new data
df_with_predictions = df_new_data.withColumn("probability", batch_inference_udf(udf_inputs))
display(df_with_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adjust output and save results to DELTA table for ad-hoc querying

# COMMAND ----------

@udf
def cast_status(input):
  if input < 0.5:
    return "healthy"
  else:
    return "damaged"

df_batch_output = df_predictions.withColumn("status", cast_status("probability"))

# COMMAND ----------

# DBTITLE 1,Save to DELTA
df_output_stats.write \
  .format("delta") \
  .mode("append") \
  .save("/LOCATION")

# COMMAND ----------

# MAGIC %md
# MAGIC ### We can now explore our prediction in a new dashboard
# MAGIC https://e2-demo-west.cloud.databricks.com/sql/dashboards/92d8ccfa-10bb-411c-b410-274b64b25520-turbine-demo-predictions?o=2556758628403379
