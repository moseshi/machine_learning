#!/usr/bin/python3
from pathlib import Path
from configparser import ConfigParser
import boto3
import os

config_path = Path("./config.ini")
error_message = """
[Error]
./config.ini not found!!!
please execute
  cp sample_config.ini config.ini
and fill the your credential
"""
assert config_path.exists(), error_message
config = ConfigParser()
config.read(str(config_path))
aws_id = config["Default"].get("aws_access_key_id", "")
aws_secret = config["Default"].get("aws_secret_access_key", "")
instance_id = config["Default"].get("instance_id", "")
region = config["Default"].get("region", "")
os.environ["AWS_ACCESS_KEY_ID"] = aws_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret
os.environ["REGION"] = region

ec2 = boto3.client("ec2", region_name=region)
instance_ids = [
    instance_id,
]
ec2.start_instances(InstanceIds=instance_ids)
