# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance (using Feature Store)
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
# MAGIC We will use sklearn's **Gradient Boosted Classifier** a famous Gradient Boosted Tree Classification library to predict which set of vibrations could be indicative of a failure.
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
# MAGIC # Exploratory Data Analysis
# MAGIC * Analyze + Clean the data
# MAGIC * Engineer features
# MAGIC * Train models
# MAGIC * Push to Model Inventory

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data from GOLD table
# MAGIC Data ready for ML

# COMMAND ----------

dataset_raw = spark.read.table("turbine_gold_for_ml")

# COMMAND ----------

# DBTITLE 1,Define potential feature column-names and output
key = "TIMESTAMP"
sensor_features = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE"]
output = ["status"]

# COMMAND ----------

# DBTITLE 1,Drop duplicate Keys/ID
dataset_cleanDF = dataset_raw.dropDuplicates(subset=[key])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import visualization libraries and display summary stats

# COMMAND ----------

# DBTITLE 1,Import libraries of choice
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Display summary statistics using Databrick's "summarize"
dbutils.data.summarize(dataset_cleanDF[sensor_features+output])

# COMMAND ----------

# DBTITLE 1,Cast to pandas dataframe given small size of dataset
df_dataset = dataset_cleanDF[sensor_features+output].toPandas() # cast to pandas
df_dataset[output] = df_dataset[output].astype("category") # cast output to category

# COMMAND ----------

# DBTITLE 1,Visualize distributions
def hide_current_axis(*args, **kwds):
  plt.gca().set_visible(False)

g = sns.pairplot(df_dataset, hue=output[0], vars=sensor_features, dropna=True, palette='Paired')
g.map_upper(hide_current_axis)

# COMMAND ----------

# DBTITLE 1,Visualize correlations
sns.heatmap(df_dataset.corr())

# COMMAND ----------

# DBTITLE 1,Count missing data
df_dataset.isna().sum().plot(kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC # Create feature tables for Feature Store
# MAGIC Use the client to create the feature tables defining meta-data such as the `database.table` which will be written to and the `key`

# COMMAND ----------

# DBTITLE 1,Select sensor features of interest
def compute_sensor_features(data, selected_features):
  """
  Select sensor data of interest + Key/ID column
  """
  
  return data.select([key] + selected_features)

# COMMAND ----------

sensor_features.remove("TORQUE")
sensor_featuresDF = compute_sensor_features(dataset_cleanDF, sensor_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate `seasonality` features
# MAGIC Since these are cyclical, transform using sine/cosine

# COMMAND ----------

from pyspark.sql.functions import month, hour, sin, cos
import numpy as np

def compute_time_cyclical_features(data):
  """
  Extract month/hour and apply cyclical calculation
  """
  return data.select(key) \
              .withColumn("Cos_Month", cos(2*np.pi*month("TIMESTAMP")/12)) \
              .withColumn("Sin_Month", sin(2*np.pi*month("TIMESTAMP")/12)) \
              .withColumn("Cos_Hour", cos(2*np.pi*hour("TIMESTAMP")/23)) \
              .withColumn("Sin_Hour", sin(2*np.pi*hour("TIMESTAMP")/23))

# COMMAND ----------

season_featuresDF = compute_time_cyclical_features(dataset_cleanDF)
display(season_featuresDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate target/response
# MAGIC Binarize output

# COMMAND ----------

# DBTITLE 0,Use for training
from pyspark.sql.functions import col

def compute_damage_response(data):
  """
  Binarize status column
  """
  return data.withColumn("Damaged", col(output[0]) == "damaged") \
             .select(key, "Damaged")

# COMMAND ----------

outputDF = compute_damage_response(dataset_cleanDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Offline database for feature store and create feature table handles
# MAGIC This is usually set-up once

# COMMAND ----------

# DBTITLE 1,Get parameters for creating personalized offline database
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("db", dbName, "DB name")

# COMMAND ----------

# DBTITLE 1,Define/Create OFFLINE database to use (if not existent already)
# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS TURBINE_FEATURES_${db}
# MAGIC   LOCATION '${path}/turbine/features';
# MAGIC 
# MAGIC USE TURBINE_FEATURES_${db};
# MAGIC DROP TABLE IF exists SENSOR_FEATURES;
# MAGIC DROP TABLE IF exists SEASON_FEATURES;

# COMMAND ----------

# DBTITLE 1,Define database.table variable names
fs_dbName = "TURBINE_FEATURES_{}".format(dbutils.widgets.get("db"))
sensor_features_table_name = f"{fs_dbName}.SENSOR_FEATURES"
season_features_table_name = f"{fs_dbName}.SEASON_FEATURES"

# COMMAND ----------

# DBTITLE 1,Use the client to create feature tables, defining metadata (e.g. database.table the feature store table will write to, and importantly, its key(s))
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

sensor_features_table = fs.create_table(
  name = sensor_features_table_name,
  keys = key,
  schema = sensor_featuresDF.schema,
  #df = sensor_featuresDF, Create and write
  description = 'Raw sensor data and Torque/Speed'
)

time_features_table = fs.create_table(
  name = season_features_table_name,
  keys = key,
  schema = season_featuresDF.schema,
  #df = season_featuresDF, Create and write
  description = 'Seasonal/Cyclical features calculated from Month and Hour'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write features to offline store

# COMMAND ----------

# DBTITLE 0,Preferred way
# MAGIC %md
# MAGIC If tables/sparkDataFrames are __already computed__ or a __streaming DataFrame__ then use `write_table` method:

# COMMAND ----------

fs.write_table(
  name = sensor_features_table_name,
  df = sensor_featuresDF,
  mode = 'overwrite' # 'merge'
)

fs.write_table(
  name = season_features_table_name,
  df = season_featuresDF,
  mode = 'overwrite' # 'merge'
)

# COMMAND ----------

# MAGIC %md
# MAGIC *Query Feature Table (since it's a Delta Table)*

# COMMAND ----------

# MAGIC %sql
# MAGIC USE turbine_features_${db};
# MAGIC SELECT * FROM sensor_features

# COMMAND ----------

# MAGIC %md 
# MAGIC # Building Model from a Feature Store: using Gradient Boosted Classifier and mlflow
# MAGIC In a feature store, we want to define the joins declaratively, because the same join will need to happen in batch inference or online serving.

# COMMAND ----------

# DBTITLE 1,Create a helper "Feature Lookup" function: this defines how features are looked up (table, feature names) and based on what key (e.g. `TIMESTAMP`)
from databricks.feature_store import FeatureLookup

def generate_lookups_per_table(table, key="TIMESTAMP"):
  return [FeatureLookup(table.name, feature_names=f, lookup_key=key) for f in fs.read_table(table.name).columns if f != key]

# COMMAND ----------

# DBTITLE 1,Get handle for feature tables
# from databricks.feature_store import FeatureStoreClient
# fs = FeatureStoreClient()
sensor_features_table = fs.get_table(sensor_features_table_name)
season_features_table = fs.get_table(season_features_table_name)

# Get list of all FeatureLookup handles
# all_feature_lookups = generate_lookups_per_table(sensor_features_table) + generate_lookups_per_table(season_features_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define training function

# COMMAND ----------

import mlflow
import mlflow.shap
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.autolog(log_input_examples=True) # Optional
label = "Damaged"
model_name = f"turbine_failure_model_{dbName}"

# Define a method for reuse later
def fit_model(model_feature_lookups, n_iter=10):

  with mlflow.start_run():
    training_set = fs.create_training_set(outputDF,
                                          model_feature_lookups,
                                          label=label,
                                          exclude_columns=key)

    # Convert to pandas Dataframe
    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop(label, axis=1)
    y = training_pd[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Add weights given class imbalance
    damage_weight = 1.0 / y_train.sum()
    healthy_weight = 1.0 / (len(y) - y_train.sum())
    sample_weight = y_train.map(lambda damaged: damage_weight if damage_weight else healthy_weight)

    # Not attempting to tune the model at all for purposes here
    gb_classifier = GradientBoostingClassifier(n_iter_no_change=n_iter)
    
    # Encode categorical cols (if any)
#     encoders = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == 'object'])])
    
    pipeline = Pipeline([("gb_classifier", gb_classifier)])
    pipeline_model = pipeline.fit(X_train, y_train, gb_classifier__sample_weight=sample_weight)

    mlflow.log_metric('test_accuracy', pipeline_model.score(X_test, y_test))
    # mlflow.shap.log_explanation(gb_classifier.predict, encoders.transform(X))

   fs.log_model(pipeline_model,
                "model",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name=model_name,
                input_example=X[:100],
                signature=infer_signature(X, y))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train an initial model (without seasonality features)

# COMMAND ----------

fit_model(generate_lookups_per_table(sensor_features_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train using seasonality features

# COMMAND ----------

fit_model(generate_lookups_per_table(sensor_features_table) + generate_lookups_per_table(season_features_table))

# COMMAND ----------

# DBTITLE 1,Get latest model programmatically
client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(model_name, stages=["None"])[0]

# COMMAND ----------

# MAGIC %md
# MAGIC # Run batch inference
# MAGIC Usinfg FS's `score_batch` method
# MAGIC 
# MAGIC __ASSUMPTION IS THAT FEATURES ARE PRE-COMPUTED__

# COMMAND ----------

# DBTITLE 1,Load latest version (can also be /Staging or /Production)
from pyspark.sql.functions import col

batch_input_df = spark.read.table(f"{dbName}.turbine_gold_for_ml")
# compute_write
with_predictions = fs.score_batch(f"models:/turbine_failure_model_{dbName}/{latest_model.version}", batch_input_df, result_type='string')
with_predictions = with_predictions.withColumn("Damaged_Predicted", col("prediction") == "True")
display(with_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Updating Feature Store Tables
# MAGIC After an initial look at feature importances, we may need to engineer more features

# COMMAND ----------

from pyspark.sql.functions import dayofweek

def compute_time_cyclical_features_v2(data):
  """
  Extract month/hour and apply cyclical calculation and if day is a week-end
  """
  return data.select(key) \
              .withColumn("Cos_Month", cos(2*np.pi*month("TIMESTAMP")/12)) \
              .withColumn("Sin_Month", sin(2*np.pi*month("TIMESTAMP")/12)) \
              .withColumn("Cos_Hour", cos(2*np.pi*hour("TIMESTAMP")/23)) \
              .withColumn("Sin_Hour", sin(2*np.pi*hour("TIMESTAMP")/23)) \
              .withColumn("is_weekend", dayofweek("TIMESTAMP").isin([1,7]).cast("int"))

# COMMAND ----------

season_v2_DF = compute_time_cyclical_features_v2(dataset_cleanDF)

# COMMAND ----------

# DBTITLE 1,Plot as bar or pie chart
display(season_v2_DF)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, with the new definitions of features in hand, `compute_and_write` can add the new features by `merge`-ing them into the feature store table.

# COMMAND ----------

fs.write_table(
  name=season_features_table_name,
  df=season_v2_DF,
  mode="merge"
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Reading feature data from a specific timestamp
# MAGIC using `as_of_delta_timestamp` parameter (__Update with your specific timestamps__)

# COMMAND ----------

season_features_df_ = fs.read_table(
  name=season_features_table_name,
  as_of_delta_timestamp="2021-10-28 21:15:00"
)

season_features_df_.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Publishing Feature Store Tables to ONLINE store
# MAGIC Publish the data in these tables to an external store (e.g. Amazon RDS MySQL, Amazon Aurora) that is more suitable for fast online lookups

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonRdsMySqlSpec

fs = FeatureStoreClient()

hostname =  dbutils.secrets.get("iot-turbine", "mysql-hostname")
port =  int(dbutils.secrets.get("iot-turbine", "mysql-port"))
user = dbutils.secrets.get("iot-turbine", "mysql-username")
password = dbutils.secrets.get("iot-turbine", "mysql-password")

online_store = AmazonRdsMySqlSpec(hostname, port, user, password)

fs.publish_table(name=sensor_features_table_name,
                 online_store=online_store,
                 streaming=False
                )

fs.publish_table(name=season_features_table_name,
                 online_store=online_store,
                 streaming=False
                )

# COMMAND ----------

# MAGIC %md
# MAGIC # More info
# MAGIC * [Supported Types](http://docs.databricks.com.s3-website-us-west-1.amazonaws.com/applications/machine-learning/feature-store/feature-tables.html#supported-data-types)
# MAGIC * [Limitations](http://docs.databricks.com.s3-website-us-west-1.amazonaws.com/applications/machine-learning/feature-store/troubleshooting-and-limitations.html#limitations)
