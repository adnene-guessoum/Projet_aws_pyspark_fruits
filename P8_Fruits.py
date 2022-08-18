# général
import pandas as pd
import numpy as np
import os
import boto3

#images
from IPython.display import Image
from PIL import Image as Image_PIL
import matplotlib.pyplot as plt
import seaborn as sns

#import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
#import sparkdl
#from sparkdl import DeepImageFeaturizer

Image_path = "Dev/Projet_8/archive/fruits-360_dataset/fruits-360/Training"

os.chdir(Image_path)

thisdir = os.getcwd()

data = []
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".jpg" in file:
            data.append(os.path.join(r,file))

print(len(data))
print(data[1])



