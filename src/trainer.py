import os

from .generators.word_generator import WordGenerator
from .processors.dummy_processor import DummyProcessor
from .data_access_managers.filesystem_manager import FileSystemManager


class Trainer:
    def __init__(self, upload_config, word_csv_path):
        self.data_manager = FileSystemManager(**upload_config)
        self.generator = WordGenerator(word_csv_path)
        self.processor = DummyProcessor()

    def submit(self, category, form_file):
        path = self.data_manager.save(form_file)
        label = self.processor(path)
        return label

    def generate(self, category=None, word=None):
        if word:
            task = self.generator.get_word(word)
        elif category:
            task = self.generator.get_category(category)
        else:
            task = self.generator.get_random()
        return task

    def get_categories(self):
        return self.generator.get_categories()
    
    def get_words(self):
        return self.generator.get_words()
