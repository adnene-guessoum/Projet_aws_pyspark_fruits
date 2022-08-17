# général
import pandas as pd
import numpy as np
import os
import time

#images
from PIL import image
from IPython.display import Image
from PIL import Image as Image_PIL
import matplotlib.pyplot as plt
import seaborn as sns

#import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
#import sparkdl
from sparkdl import DeepImageFeaturizer

Image_path = "..\S3_test_P8"

list_img = [file for file in os.listdir(Image_path)]
print(len(list_img))




