import os
import boto3
from pathlib import Path
from datetime import datetime

start = datetime.now()

from example_gan_cifar10 import score, models

end = datetime.now()

result_dir = "./result/"

result_dir_path = Path(result_dir)
if not result_dir_path.exists():
    result_dir_path.mkdir(0o775, True)

for model_name, model_obj in models.items():
    model_obj.save("{}/{}".format(result_dir, model_name))
result_path = "result.txt".format(datetime.now())
with open("{}/{}".format(result_dir, result_path), "w") as f:
    f.write("Test loss:{}".format(score[0]))
    f.write("Test accuracy:{}".format(score[1]))
    f.write("Run time: {}".format((end - start).seconds))
s3 = boto3.resource("s3")
bucket_name = "machine-learning"
bucket = s3.Bucket(bucket_name)
date = datetime.now()
upload_dir = date.strftime("%y-%m-%d-%H-%M-%S")
for p in Path("./result").iterdir():
    if p.is_dir():
        continue
    bucket.upload(model_path,"{}/{}".format(upload_dir, str(p)))
    p.unlink()
