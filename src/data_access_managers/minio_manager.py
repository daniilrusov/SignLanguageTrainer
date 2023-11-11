from werkzeug.utils import secure_filename
from minio import Minio
import os
import time
import cv2
import numpy as np
import base64


class MinioManager:
    def __init__(self, MINIO_API_HOST, ACCESS_KEY, SECRET_KEY, BUCKET_NAME):
        self.minio = Minio(MINIO_API_HOST, ACCESS_KEY, SECRET_KEY, secure=False)
        self.BUCKET_NAME = BUCKET_NAME
        found = self.minio.bucket_exists(BUCKET_NAME)
        if not found:
            self.minio.make_bucket(BUCKET_NAME)
    
    def save(self, images):
        video_name = f"video{time.time()}.mp4"
        output = os.path.join(self.folder, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(output, fourcc, 20.0, (638, 478))
        print(len(images))
        for image in images:
            encoded_data = image.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            out.write(frame) # Write out frame to video
        out.release()
        file = os.open(output)
        self.minio.put_object(self.BUCKET_NAME, video_name, file, os.stat(output).st_size)
        url = self.minio.presigned_get_object(self.BUCKET_NAME, video_name)
        return url, video_name
    
    def delete(self, path):
        self.minio.remove_object(self.BUCKET_NAME, path)