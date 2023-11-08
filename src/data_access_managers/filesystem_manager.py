from werkzeug.utils import secure_filename
import os


class FileSystemManager:
    def __init__(self, folder):
        self.folder = folder
    
    def save(self, file):
        filename = secure_filename(file.filename)
        full_path = os.path.join(self.folder, filename)
        file.save(full_path)
        return full_path
    
    def get(self, filename):
        # load video from filesystem
        pass 