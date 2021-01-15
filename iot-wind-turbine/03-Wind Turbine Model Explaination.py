# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this example, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/turbine/turbine_flow.png" />
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

# MAGIC %md ### 

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: Workflows with Pyspark.ML Pipeline

# COMMAND ----------

# DBTITLE 1,Build Training and Test dataset
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
dataset = spark.read.table("quentin.turbine_gold").orderBy(rand())
train, test = dataset.limit(1000000).randomSplit([0.8, 0.2])

# COMMAND ----------

# DBTITLE 1,Train our model using a GBT
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir
import mlflow.spark
import mlflow
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


with mlflow.start_run():
  #the source table will automatically be logged to mlflow
  #mlflow.spark.autolog()
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5]).build()

  ev = BinaryClassificationEvaluator()
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=ev, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="STATUS", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(train)
  
  mlflow.spark.log_model(pipelineTrained, "turbine_gbt")
  mlflow.set_tag("model", "turbine_gbt")
  predictions = pipelineTrained.transform(test)

  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  mlflow.log_metric("precision", metrics.precision(1.0))
  mlflow.log_metric("recall", metrics.recall(1.0))
  mlflow.log_metric("f1", metrics.fMeasure(1.0))
  AUROC = ev.evaluate(predictions)
  mlflow.log_metric("AUROC", AUROC)
  
  #Add confusion matrix to the model:
  with TempDir() as tmp_dir:
    labels = pipelineTrained.stages[2].labels
    sn.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(tmp_dir.path()+"/confusion_matrix.png")
    mlflow.log_artifact(tmp_dir.path()+"/confusion_matrix.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Explainability
# MAGIC Our spark model comes with a basic feature importance metric we can use to have a first understanding of our mode:

# COMMAND ----------

bestModel = pipelineTrained.stages[-1:][0].bestModel
# convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")
display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

# MAGIC %md #### Explaining our model with SHAP
# MAGIC Our model feature importance are quite limited (we can't explain a single prediction) and can lead to surprising result. You can have a look to [Scott Lundberg blog post](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) for more details

# COMMAND ----------

import shap
import numpy as np
#We'll need to add shap bundle js to display nice graph
with open(shap.__file__[:shap.__file__.rfind('/')]+"/plots/resources/bundle.js", 'r') as file:
   shap_bundle_js = '<script type="text/javascript">'+file.read()+'</script>'
#Build our explainer    
explainer = shap.TreeExplainer(bestModel)

#Let's draw the shap value (~force) of each feature
X = dataset.select(featureCols).limit(1000).toPandas()
shap_values = explainer.shap_values(X, check_additivity=False)
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:6], ["Mean |SHAP|", "Column"]))

# COMMAND ----------

# MAGIC %md #### Analyzing a specific value
# MAGIC Using shap, we can understand how our model is behaving for a specific row. Let's analyze the importance of each feature for the first row of our dataset.
# MAGIC 
# MAGIC The following explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. 
# MAGIC 
# MAGIC Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.

# COMMAND ----------

plot_html = shap.force_plot(explainer.expected_value, shap_values[884,:], X.iloc[884,:], feature_names=X.columns)
displayHTML(shap_bundle_js + plot_html.html())

# COMMAND ----------

# MAGIC %md 
# MAGIC If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset:

# COMMAND ----------

X = dataset.select(featureCols).limit(1000).toPandas()
shap_values = explainer.shap_values(X, check_additivity=False)
plot_html = shap.force_plot(explainer.expected_value, shap_values, X)
displayHTML(shap_bundle_js + plot_html.data)

# COMMAND ----------

# MAGIC %md #### Overview of all features
# MAGIC To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). 
# MAGIC 
# MAGIC This reveals for example that big negative AN5 strongly influence the prediction for a damaged turbine (negative SHAP value, the purple = data around 0 dot are stacked on the left, and data with high value (positive or negative) on the right)

# COMMAND ----------

# summarize the effects of all the features
shap.summary_plot(shap_values, X)

# COMMAND ----------

# MAGIC %md To understand how a single feature effects the output of the model we can plot the SHAP value of that feature vs. the value of the feature for all the examples in a dataset. 
# MAGIC 
# MAGIC Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in turbine health as AN9 changes. Vertical dispersion at a single value of AN9 represents interaction effects with other features. 
# MAGIC 
# MAGIC To help reveal these interactions dependence_plot can selects another feature for coloring. In this case, we realize that AN9 and AN3 are liked: purple values (0) are stacked where AN3=3 (you can try with interaction_index=None to remove color).
# MAGIC 
# MAGIC We clearly see from this that rows having a AN3 close to 0 (no vibration) have a low SHAP value (healthy).

# COMMAND ----------

shap.dependence_plot("AN3", shap_values, X, interaction_index="AN9")

# COMMAND ----------

# MAGIC %md #### Computing SHAP values on the entier dataset:
# MAGIC These graph are great to understand the model against a subset of data. If we want to to further analyze based on the shap values on millions on rows, we can use spark to compute the shap values.
# MAGIC 
# MAGIC We can use spark 3 `mapInPandas` function, or create a `@pandas_udf`:

# COMMAND ----------

import pandas as pd
#Note: requires the last shap version, see https://github.com/slundberg/shap/issues/38
features = dataset.select(featureCols)
def compute_shap_values(iterator):
  for X in iterator:
    yield pd.DataFrame(explainer.shap_values(X, check_additivity=False))

display(features.mapInPandas(compute_shap_values, schema=", ".join([x+"_shap_value float" for x in features.columns])))