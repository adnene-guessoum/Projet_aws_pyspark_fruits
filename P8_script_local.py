import os
import boto3
import sparkdl
import pyspark.sql.functions as F

#spark
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions
from pyspark.sql.types import *

### Read images and vectorize
from pyspark.ml.image import ImageSchema
from pyspark.ml.feature import PCA
import time

# necessary import 
from functools import reduce
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


run_local = False

# create a spark session
spark = SparkSession.builder.master('local[*]').appName('P8-fruits').getOrCreate()
sc = spark.sparkContext.getOrCreate()

#le label est le nom du dossier qui contient l'image :
def find_label(path):
        return path.split('/')[-2]

#read images locally or from S3 :
def load_image_into_pspark_df(run_local = False):
    if run_local :
        s3 = boto3.client("s3")
        AWS_REGION = "eu-west-1"
        bucket = 'echantillon-img'
        location = {'LocationConstraint': AWS_REGION}
    else:
        image_path = "/home/adnene/Dev/Projet-8/echantillon-img/*/"

    images_df = spark.read.format("image").load(image_path)
    images_df = images_df.withColumn("chemin", F.input_file_name())

    udf_categorie = F.udf(find_label, StringType())
    images_df = images_df.withColumn('categorie', udf_categorie('chemin'))
    
    return images_df

def extract_features_images(df):
    
    featurizer = sparkdl.DeepImageFeaturizer(inputCol="image",
    outputCol="processed_image", modelName="ResNet50" )
    
    processed_df = featurizer.transform(df).select(['path', 'label', 'processed_image'])
    del featurizer
    return processed_df

def dim_reduction(df):
    acp = PCA(k=6, inputCol="processed_image", outputCol="reduced_features")
    
    return complete_df


if __name__ == "__main__":
    pass

'''
#path to images:
if From_S3 :
    #utiliser les images sur s3:
    s3 = boto3.client("s3")
    AWS_REGION = "eu-west-1"
    bucket = 'echantillon-img'
    location = {'LocationConstraint': AWS_REGION}
    
else:
    local_image_path = r"Dev/Projet-8/echantillon-img"

def loading_images_spark_df(path, From_S3 = False):
    if From_S3 :
        pass

    else:
        dir_list = os.listdir(path)
        for i in dir_list:
            im_path = os.path.join(path, i)

    image_df = spark.read.format("image").load(DATA_PATH)

    return 

Im_df = loading_images_spark_df(local_image_path)
Im_df.show()












  all_images = s3.list_objects(Bucket = bucket) 
'''