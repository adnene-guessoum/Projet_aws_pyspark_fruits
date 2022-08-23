


import pyarrow.parquet as pq
import s3fs
s3 = s3fs.S3FileSystem()

results_df = pq.ParquetDataset('s3://echantillon-img/data-output/results', filesystem=s3).read_pandas().to_pandas()


s3 = boto3.resource('s3')
bucket = s3.Bucket('echantillon-img')

images = []
for obj in bucket.objects.all():
    path = obj.key
    label = obj.key.split('/')[-2]
    body = obj.get()['Body'].read()
    images.append([path, label, body])
