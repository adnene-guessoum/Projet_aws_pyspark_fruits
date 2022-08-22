 
 
import boto3

        
s3 = boto3.client("s3")
AWS_REGION = "eu-west-1"
bucket = 'echantillon-img'
location = {'LocationConstraint': AWS_REGION}