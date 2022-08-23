import boto3
import pandas as pd
from PIL import Image
import numpy as np
import io

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import PCA , StandardScaler
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.master('local[*]').appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()
sc.setLogLevel('ERROR')

import json

with open('/home/adnene/Dev/Projet-8/awsaccesskeys.txt', 'r') as f:
    datastore = json.load(f)

hadoopConf = sc._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.access.key", datastore['ID'])
hadoopConf.set("fs.s3a.secret.key", datastore['key'])

'''
Définir les fonctions utiles :
'''

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
    return im.select(col("path"), col("label"), featurize_udf("content").alias("features"))

def dimension_red(df):
    #spark ne normalise pas les données automatiquement :
    df_index = df.select("*").withColumn("id", monotonically_increasing_id())
    features = df_index.select(['id', 'features'])
    features_RDD = features.rdd.map(lambda x : (x['id'], Vectors.dense(x['features']))).toDF(['id', 'features'])
    
    #normalisation
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(features_RDD)
    scaledData = scalerModel.transform(features_RDD)

    #pca
    acp = PCA(k=6, inputCol=scaler.getOutputCol(), outputCol="pca_features")
    model_acp = acp.fit(scaledData)
    reduced_feature = model_acp.transform(scaledData)

    #construction df final
    complete_df = df_index.join(reduced_feature.select(['id', 'pca_features']), on=['id']).drop('id')
  
    del acp
    del model_acp
    return complete_df
    
def write_results_into_parquet(df, path_results):
        df.write.mode('overwrite').save(path_results)
        print("loading Done")

if __name__ == "__main__":

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('echantillon-img')

    images = []
    for obj in bucket.objects.all():
      if obj.key.endswith('.jpg'):
        path = obj.key
        label = obj.key.split('/')[-2]
        body = obj.get()['Body'].read()
        images.append([path, label, body])

    images = pd.DataFrame(images, columns= ['path', 'label', 'content'])
    images = spark.createDataFrame(images)

    model = ResNet50(include_top=False)
    bc_model_weights = sc.broadcast(model.get_weights())
    
    processed = extract_features_images(images)
    results = dimension_red(processed)
    results.show()
    
    #write_results_into_parquet(results, 's3a://echantillon-img/data-output/results')



