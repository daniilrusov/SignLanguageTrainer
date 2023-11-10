from werkzeug.utils import secure_filename
import os
import time
import cv2
import numpy as np
import base64


class FileSystemManager:
    def __init__(self, folder):
        self.folder = folder
    
    #def save(self, file):
    #    filename = secure_filename(file.filename)
    #    full_path = os.path.join(self.folder, filename)
    #    file.save(full_path)
    #    return full_path
    
    def save(self, images):
        output = os.path.join(self.folder, f"video{time.time()}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(output, fourcc, 20.0, (638, 478))
        print(len(images))
        for image in images:
            encoded_data = image.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            out.write(frame) # Write out frame to video
        out.release()
        return output

    def delete(self, path):
        if os.path.exists(path):
            os.remove(path)