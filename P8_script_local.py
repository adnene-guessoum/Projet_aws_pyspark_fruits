import boto3

#spark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

### Read images and vectorize
from pyspark.ml.image import ImageSchema
from pyspark.ml.feature import PCA
from pyspark.ml.feature import StandardScaler
import time

# featurizer import 
from functools import reduce
import tensorflow as tf
from keras.applications import VGG16

import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import monotonically_increasing_id 

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors


run_s3 = False

# create a spark session
spark = SparkSession.builder.master('local[*]').appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()

model = ResNet50(include_top=False)
bc_model_weights = sc.broadcast(model.get_weights())

#le label est le nom du dossier qui contient l'image :
def find_label(path):
        return path.split('/')[-2]

#read images locally or from S3 :
def load_image_into_pspark_df(run_s3 = False):
    if run_s3 :
        s3 = boto3.client("s3")
        AWS_REGION = "eu-west-1"
        bucket = 'echantillon-img'
        location = {'LocationConstraint': AWS_REGION}
    else:
        image_path = "/home/adnene/Dev/Projet-8/echantillon-img"

    images = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(image_path)

    udf_categorie = F.udf(find_label, StringType())
    images = images.withColumn('label', udf_categorie('path'))

    return images


#Extraction des features images :

#procédure décrite dans https://lytix.be/transfer-learning-in-spark-for-image-recognition/ 
#et dans documentation databricks :

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model_res = ResNet50(weights=None, include_top=False)
  model_res.set_weights(bc_model_weights.value)
  return model_res

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  print(preds)
  output = [p.flatten() for p in preds]
  return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

def extract_features_images(im):
    return im.repartition(4).select(col("path"), col("label"), featurize_udf("content").alias("features"))

def dimension_red(df):
    #spark ne normalise pas les données automatiquement :
    df_index = df.select("*").withColumn("id", monotonically_increasing_id())
    features = df_index.select(['id', 'features'])
    features_RDD = features.rdd.map(lambda x : (x['id'], Vectors.dense(x['features']))).toDF(['id', 'features'])
   
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(features_RDD)
    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(features_RDD)

    acp = PCA(k=6, inputCol=scaler.getOutputCol(), outputCol="pca_features")
    model_acp = acp.fit(scaledData)
    reduced_feature = model_acp.transform(scaledData)

    complete_df = df_index.join(reduced_feature.select(['id', 'pca_features']), on=['id']).drop('id')
  
    del acp
    del model_acp
    return complete_df
    
def write_results_into_parquet(df, path_results):
        df.write.format("parquet").save(path_results)
        print("loading Done")


if __name__ == "__main__":
    images = load_image_into_pspark_df()
    processed = extract_features_images(images)
    results = dimension_red(processed)
    results.show()
    write_results_into_parquet(results, r"/home/adnene/Dev/Projet-8/echantillon-img/results")

