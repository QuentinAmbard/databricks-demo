# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Search best hyper parameters with HyperOpt (Bayesian optimization) accross multiple nodes
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/bayesian-model.png" style="height: 330px"/></div>
# MAGIC 
# MAGIC This model is a good start, but now we want to try multiple hyper-parameter to see how it behaves.
# MAGIC 
# MAGIC GridSearch could be a good way to do it, but not very efficient when the parameter dimension increase and the model is getting slow to train due to a massive amount of data.
# MAGIC 
# MAGIC HyperOpt search accross your parameter space for the minimum loss of your model, using Baysian optimization instead of a random walk

# COMMAND ----------

# DBTITLE 1,We'll use Prophet to run our prediction. Let's install it in a new conda env
# MAGIC %conda install -c conda-forge fbprophet

# COMMAND ----------

# DBTITLE 1,Data initialisation (Make sure prophet is available and installed with conda)
# MAGIC %run ./resources/01-setup-prophet $reset_all_data=$reset_all_data

# COMMAND ----------

# DBTITLE 0,Define search space
param_hyperopt = {'interval_width': hp.uniform('randomForest__n_estimators', 0.5, 0.95), 
                  'growth': 'linear',         
                  'daily_seasonality': hp.choice('daily_seasonality', [True, False]),
                  'weekly_seasonality': hp.choice('weekly_seasonality', [True, False]),
                  'yearly_seasonality': hp.choice('yearly_seasonality', [True, False]),
                  'seasonality_mode': 'multiplicative'}

# COMMAND ----------

# DBTITLE 0,Define the function we want to minimize
history_sample = spark.read.table("sales_history").where(col("date") >= "2015-01-01").sample(fraction=0.01, seed=123)
history_pd = history_sample.toPandas().rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]

# Function to train prophet model and forecast using hyperopt params
def train_prophet_hyperopt(param_hyperopt):
  model = define_prophet_model(param_hyperopt)
  model.fit(history_pd)
  future_pd = make_predictions(model, 30)
  forecast_pd = model.predict(future_pd)
  metrics = evaluate_metrics(model)
  rmse = metrics.loc[0,'rmse']

  return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'param_hyperopt': param_hyperopt}

# Function to run hyperopt param searh trails 
def run_prophet_hyperopt(trials = Trials()):
  
  with mlflow.start_run(run_name="Hyperopt Prophet Run", nested = True):   
    argmin = fmin(train_prophet_hyperopt, param_hyperopt, algo=tpe.suggest, max_evals=12, show_progressbar=True, trials = trials)
   
    rmse = trials.best_trial['result']['loss']
    #########################################
    # Logging best param & model to MLFlow  #
    #########################################
    mlflow.pyfunc.log_model("model", conda_env=conda_env, python_model=FbProphetWrapper(trials.best_trial['result']['model']))  
    for key, value in space_eval(param_hyperopt, argmin).items():
      mlflow.log_param(key, value)
    mlflow.set_tag("model", "hyperopt_sales_forecast")    
    mlflow.log_metric("rmse", rmse)
  
    return rmse


# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/hyperopt-spark.png" style="height: 300px; margin-left:20px"/></div>
# MAGIC #### Distribute HyperOpt accross multiple nodes
# MAGIC HyperOpt is ready to be used with your spark cluster and can automatomatically distribute the workload accross multiple instances.
# MAGIC 
# MAGIC Spark Hyperopt also automatically log all trials to MLFLow!

# COMMAND ----------

# DBTITLE 0,Run hyperopt with spark
run_prophet_hyperopt(trials = SparkTrials())

# COMMAND ----------

# MAGIC %md ## Generating Store-Item Level Forecasts in Parallel
# MAGIC 
# MAGIC Our model is great, but having a generic model for all items across all stores isn't accurate and won't help forecasting item-level sales.
# MAGIC 
# MAGIC ![](https://github.com/HimanshuAroraDb/Images/blob/master/multimodel.png?raw=true)
# MAGIC 
# MAGIC 
# MAGIC Leveraging Spark and Databricks, we can easily solve this problem.  Instead of iterating over the set of store-item combinations, we will simply group our data by store and item, forcing store-item combinations to be partitioned across the resources in our cluster. To each store-item grouping, we will apply a function, similar to what we did before, to generate a forecast for each combination. The result will be a unified dataset, addressable as a Spark DataFrame.
# MAGIC 
# MAGIC To get us started, let's re-write our forecast-generating function so that it may be applied to a Spark DataFrame. What you'll notice is that we are defining this function as a [pandas UDF](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) which enables the efficient application of pandas functionality to grouped data in a Spark DataFrame. 
# MAGIC 
# MAGIC Despite the slightly different function signature (which requires us to pre-define the structure of the pandas DataFrame that this function will produce), the internal logic is largely the same as the previous function:

# COMMAND ----------

# DBTITLE 1,Define Training & Forecasting Function for Spark
# get forecast
@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP )
def get_forecast_spark(keys, grouped_pd):
  
  # drop nan records
  grouped_pd = grouped_pd.dropna()

  # identify store and item
  store = keys[0]
  item = keys[1]
  days_to_forecast = keys[2]

  # configure model
  model = define_prophet_model(params)

  # train model
  model.fit( grouped_pd.rename(columns={'date':'ds', 'sales':'y'})[['ds', 'y']]  )

  # make forecast
  future_pd = make_predictions(model, days_to_forecast)

  # retrieve forecast
  forecast_pd = model.predict( future_pd )

  # assign store and item to group results
  forecast_pd['store']=store
  forecast_pd['item']=item

  # return results
  return forecast_pd[[c.name for c in result_schema]]

# COMMAND ----------

# MAGIC %md With our function defined, we can now group our data and apply the function to each group to generate a store-item forecast:

# COMMAND ----------

# DBTITLE 1,Train & Forecast in Parallel with Spark
# generate forecasts
# the days_to_forecast field is used to overcome inability to pass params to pandas udf
store_item_accum_spark = spark.read.table("sales_history").groupBy('store', 'item', lit(30).alias('days_to_forecast')).apply(get_forecast_spark).cache()
# save forecast as view
store_item_accum_spark.createOrReplaceTempView('store_item_forecast')

# display some results on screen
display(store_item_accum_spark)

# COMMAND ----------

# DBTITLE 1,Store specific forecast for item1
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   store,
# MAGIC   ds,
# MAGIC   yhat
# MAGIC FROM store_item_forecast
# MAGIC WHERE item=1 AND ds >= '2018-01-01'
# MAGIC ORDER BY store, ds

# COMMAND ----------

# MAGIC %md
# MAGIC ### What we've done  
# MAGIC 1. Expored reliability features of Delta lake
# MAGIC 1. Data exploration & data analysis using sql and inbuilt graphes
# MAGIC 1. Build a model using Prophet and MLflow for tracking
# MAGIC 1. Save this model on MLFLow registry
# MAGIC 1. Run the same operation, training 1 model per item per shop at scale using Spark 
# MAGIC 
# MAGIC ### What's next ?
# MAGIC 1. Track our data & model performance using a SQL Analytics Dashboard
