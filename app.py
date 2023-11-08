from flask import Flask, render_template, request, url_for, flash, redirect
import cv2
import os
from src.trainer import Trainer

UPLOAD_PATH = 'uploads_folder'
WORDS_CSV = r'words.csv'

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
    file = request.files['video']
    category = request.values['category']
    trainer.submit(category, file)
    return "dummy"

@app.route('/getTask/')
def getTask():
    word = request.args.get('word')
    word = word if word != 'null' else None
    category = request.args.get('category')
    category = category if category != 'null' else None

    task = trainer.generate(word=word, category=category)
    return task._asdict()
