from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
import cv2
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
UPLOAD_PATH = 'uploads_folder'
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit/', methods=['POST'])
def submit():
    print(request.values['word'])
    print(request)
    file = request.files['video']
    print(file)
    filename = secure_filename(file.filename)
    print(filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return "correct"
