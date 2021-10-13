# Databricks notebook source
#dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"])

# COMMAND ----------

# MAGIC %run ./00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import mlflow 
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import json


# COMMAND ----------

reset_all = dbutils.widgets.get("reset_all_data") == "true"
if not spark._jsparkSession.catalog().tableExists('customer_segmentation') or reset_all:
  print("loading segmentation table")
  spark.read.format("parquet").load("/mnt/quentin-demo-resources/retail/segmentation").write.mode("overwrite").saveAsTable("customer_segmentation")
