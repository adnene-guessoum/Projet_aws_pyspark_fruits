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
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from tensorflow.keras.models import load_model

from pyspark.ml.image import ImageSchema

#load environment and utilities from s3
spark = SparkSession.builder.appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()
sc.setLogLevel('ERROR')

S3_INPUT = "s3a://echantillon-img/Images/*"
S3_OUTPUT = "s3a://echantillon-img/data-output/results"

images = spark.read.format("binaryFile").load(S3_INPUT)
images_df = images.select("path", "content" )

get_label = udf(lambda x: x.split('/')[-2])
images_df = images_df.select("*").withColumn("label", get_label(images_df.path))
#images = spark.createDataFrame(images)
images_df.show()

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  client = boto3.client('s3')
  client.download_file("echantillon-img",
                      'model/model_res.h5',
                      'my_model.h5')
# returns a compiled model
  model_load = load_model("my_model.h5")
  return model_load

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


features_df = images_df.select(col("path"), col("label"), featurize_udf("content").alias("features"))

features_df.show()

#partie dimension reduction :

#conserver un id par lignes pour joindre les df à la fin
df_index = features_df.select("*").withColumn("id", monotonically_increasing_id())
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

