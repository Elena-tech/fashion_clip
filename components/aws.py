import json
import os

import boto3
from botocore.exceptions import ClientError


class RunRekognition:
    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION")
        self.bucketname = os.getenv("BUCKET")
        self.confidencestr = os.getenv("CONFIDENCE")
        self.confidence = float(self.confidencestr)
        self.boto3client = boto3.client(
            "rekognition",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        )
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        )

    def get_labels(self, file_dir):
        with open(file_dir, "rb") as image:
            response = self.boto3client.detect_labels(Image={"Bytes": image.read()})

        image_labels = []
        for label in response["Labels"]:
            if label["Confidence"] > self.confidence:
                image_labels.append(label["Name"].lower())
        # Generate a prompt by concatenating the image labels
        return ", ".join(image_labels)

    def get_text(self, file_dir):
        with open(file_dir, "rb") as image:
            response = self.boto3client.detect_text(Image={"Bytes": image.read()})

        image_text = []
        for text in response["TextDetections"]:
            if text["Confidence"] > self.confidence:
                image_text.append(text["DetectedText"].lower())
        # Generate a prompt by concatenating the image labels
        return ", ".join(image_text)

    def get_celeb(self, file_dir):
        with open(file_dir, "rb") as image:
            response = self.boto3client.recognize_celebrities(
                Image={"Bytes": image.read()}
            )
        print(response)
        image_text = []
        for text in response["CelebrityFaces"]:
            if text["MatchConfidence"] > self.confidence:
                image_text.append(text["Name"].lower())
        # Generate a prompt by concatenating the image labels
        return ", ".join(image_text)

    def bucket_exists(self):
        try:
            # 'HeadBucket' only retrieves metadata about the bucket, not its contents
            self.s3.head_bucket(Bucket=self.bucketname)
            return True
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                return False
            elif error_code == 403:
                print(
                    f"Permissions denied for bucket: {self.bucketname}. Assuming it exists."
                )
                return True
            else:
                # Some other unexpected error occurred
                raise

    def create_s3_bucket(self):
        image_uri = None
        if not self.bucket_exists():
            location = {"LocationConstraint": self.region} if self.region else {}
            image_uri = self.s3.create_bucket(
                Bucket=self.bucketname, CreateBucketConfiguration=location
            )
            # self.make_bucket_public(s3)
        return image_uri

    def make_bucket_public(self):
        public_bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadGetObject",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucketname}/*",
                }
            ],
        }
        self.s3.put_bucket_policy(
            Bucket=self.bucketname, Policy=json.dumps(public_bucket_policy)
        )

    def upload_to_bucket(self, blob_name, path_to_file):

        self.create_s3_bucket()
        try:
            self.s3.upload_file(path_to_file, self.bucketname, blob_name)
            print(f"Image uploaded successfully to s3://{self.bucketname}/{blob_name}")

            # Construct the public URL for the uploaded object
            # Note: This assumes the object is publicly accessible. If it's not, accessing the URL will result in a permission error.
            url = f"https://{self.bucketname}.s3.amazonaws.com/{blob_name}"
            return url
        except Exception as e:
            print(f"Error uploading {path_to_file} to {self.bucketname}. Error: {e}")
            return None
