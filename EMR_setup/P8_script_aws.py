import boto3
import pandas as pd
from PIL import Image
import numpy as np
import io
import sys
sys.setrecursionlimit(3000)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import PCA , StandardScaler
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.ml.linalg import Vectors
from tensorflow.keras.models import load_model

spark = SparkSession.builder.appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()
sc.setLogLevel('ERROR')


'''import json

with open('/home/adnene/Dev/Projet-8/awsaccesskeys.txt', 'r') as f:
    datastore = json.load(f)

hadoopConf = sc._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.access.key", datastore['ID'])
hadoopConf.set("fs.s3a.secret.key", datastore['key'])'''

'''
Définir les fonctions utiles :
'''

#Extraction des features images :
#procédure décrite dans https://lytix.be/transfer-learning-in-spark-for-image-recognition/ 
#et dans documentation databricks :

"""def model_fn():
  
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  
  model_res = model_load
  
  '''ResNet50(weights=None, include_top=False)
  model_res.set_weights(bc_model_weights.value)'''
  return model_res"""

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_series(content_series_iter):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  model_load = load_model('my_model.h5', compile=False)
  
  for content_series in content_series_iter:
    input = np.stack(content_series.map(preprocess))
    preds = model_load.predict(input)
    output = [p.flatten() for p in preds]
    yield pd.Series(output)
    
def extract_features_images(im):
  im_feat = im.select("*").withColumn("features", featurize_series(im.content))
  return im_feat

def do_indexation(df):
  return df.select("*").withColumn("id", monotonically_increasing_id())

def transform_dataframe(df_index):
  #spark ne normalise pas les données automatiquement :
  features = df_index.select(['id', 'features'])
  features_mapped = features.rdd.map(lambda x : (x['id'], Vectors.dense(x['features']))).toDF(['id', 'features'])
  return features_mapped

def scaling_df(df):
  #normalisation
  scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
  scalerModel = scaler.fit(df)
  scaledData = scalerModel.transform(df)
  return scaledData

def do_pca(df):
  #pca
  acp = PCA(k=6, inputCol="scaledFeatures", outputCol="pca_features")
  model_acp = acp.fit(df)
  reduced_feature = model_acp.transform(df)

  #construction df final
  complete_df = df_index.join(reduced_feature.select(['id', 'pca_features']), on=['id']).drop('id')
  return complete_df
    
def write_results_into_parquet(df, path_results):
        df.write.mode('overwrite').save(path_results)
        print("loading Done")

if __name__ == "__main__":

    S3_INPUT = "s3a://echantillon-img"
    S3_OUTPUT = "s3a://echantillon-img/data-output/results"

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

    '''model = ResNet50(include_top=False)
    bc_model_weights = sc.broadcast(model.get_weights())'''
    
    client = boto3.client('s3')
    client.download_file("echantillon-img",
                        'model/model_res.h5',
                        'my_model.h5')
    # returns a compiled model
    # identical to the previous one
    model_load = load_model('my_model.h5', compile=False)

    processed = extract_features_images(images)

    df_index = do_indexation(processed)
    
    df = transform_dataframe(df_index)

    scaledData = scaling_df(df)

    results = do_pca(scaledData)

    results.show()

    write_results_into_parquet(results, S3_OUTPUT)



