import boto3
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import load_model
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()

model = ResNet50(include_top=False)
bc_model_weights = sc.broadcast(model.get_weights())

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model_res = ResNet50(weights=None, include_top=False)
  model_res.set_weights(bc_model_weights.value)
  return model_res

model = model_fn()

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
client = boto3.client('s3')
client.upload_file(Filename='my_model.h5',
                  Bucket="echantillon-img",
                  Key='model/model_res.h5')

del model  # deletes the existing model

print("Done")

'''client = boto3.client('s3')
client.download_file("echantillon-img",
                     'model/model_res.h5',
                     'my_model.h5')
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')'''