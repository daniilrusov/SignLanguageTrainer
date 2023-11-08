from werkzeug.utils import secure_filename
from minio import Minio


class MinioManager:
    def __init__(self, path, access_key, secret_key, bucket):
        self.bucket = bucket
        self.minio = Minio(path, access_key=access_key, secret_key=secret_key)
    
    def save(self, file):
        # save to minio
        pass
    
    def get(self, filename):
        # load video from minio
        pass