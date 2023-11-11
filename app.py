from flask import Flask, render_template, request, url_for, flash, redirect
from src.trainer import Trainer
import os

UPLOAD_PATH = 'uploads_folder'
WORDS_CSV = r'words.csv'

"""
ACCESS_KEY = os.environ.get("MINIO_ROOT_USER")
SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD")
BUCKET_NAME = os.environ.get("MINIO_BUCKET")
MINIO_API_HOST = os.environ.get("MINIO_ENDPOINT")

upload_config= {"MINIO_API_HOST": MINIO_API_HOST, 
                "ACCESS_KEY": ACCESS_KEY,
                "SECRET_KEY": SECRET_KEY,
                "BUCKET_NAME": BUCKET_NAME}
"""

upload_config = {'folder': UPLOAD_PATH}

trainer = Trainer(upload_config, WORDS_CSV)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    categories = trainer.get_categories()
    words = trainer.get_words()
    return render_template('index.html', categories=categories, words=words)

@app.route('/submit/', methods=['POST'])
def submit():
    request_json = request.get_json()
    #print(request_json)
    images = request_json['images']
    duration = request_json['duration']
    category = request_json['category']

    #file = request.form.getlist('images[]')
    #duration = request.values['duration']
    print(len(images))
    print(duration)
    #category = request.values['category']
    label = trainer.submit(category, images)
    #label = "asdasd"
    return label

@app.route('/getTask/')
def getTask():
    word = request.args.get('word')
    word = word if word != 'null' else None
    category = request.args.get('category')
    category = category if category != 'null' else None

    task = trainer.generate(word=word, category=category)
    return task._asdict()

if __name__ == "__main__":
    app.run(host='0.0.0.0')