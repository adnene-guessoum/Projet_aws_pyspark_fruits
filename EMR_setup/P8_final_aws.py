import boto3
import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow
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

#load environment and utilities from s3
spark = SparkSession.builder.master("local[*]").appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()
sc.setLogLevel('ERROR')

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

client = boto3.client('s3')
client.download_file("echantillon-img",
                    'model/model_res.h5',
                    'my_model.h5')
# returns a compiled model
model_load = load_model('my_model.h5', compile=False)

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
    

processed = images.select("*").withColumn("features", featurize_series(images.content))

#acp dimension reduction :
df_index = processed.select("*").withColumn("id", monotonically_increasing_id())
features = df_index.select(['id', 'features']).rdd
features_RDD = features.map(lambda x : (x['id'], Vectors.dense(x['features']))).toDF(['id', 'features'])

#normalisation
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(features_RDD)
scaledData = scalerModel.transform(features_RDD)

#pca
acp = PCA(k=6, inputCol=scaler.getOutputCol(), outputCol="pca_features")
model_acp = acp.fit(scaledData)
reduced_feature = model_acp.transform(scaledData)

#construction df final
results = df_index.join(reduced_feature.select(['id', 'pca_features']), on=['id']).drop('id')

results.show()

results.write.mode('overwrite').save(S3_OUTPUT)