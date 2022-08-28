
import boto3
import pandas as pd
import numpy as np

import pyarrow.parquet as pq
import s3fs
s3 = s3fs.S3FileSystem()

results = pq.ParquetDataset('s3://echantillon-img/data-output/results', filesystem=s3)
results_df = pd.read_parquet(results, engine='pyarrow')

print(results_df["pca_features"])

'''s3 = boto3.resource('s3')
bucket = s3.Bucket('echantillon-img')

images = []
for obj in bucket.objects.all():
    path = obj.key
    label = obj.key.split('/')[-2]
    body = obj.get()['Body'].read()
    images.append([path, label, body])
'''