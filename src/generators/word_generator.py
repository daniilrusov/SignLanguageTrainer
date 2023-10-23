import pandas as pd

class WordGenerator:
    def __init__(self, words_path):
        self.words_df = pd.read_csv(words_path) # df with cols: word, guide_path, category
    
    def get_random(self):
        pass

    def get_category(self, category):
        pass

    def get_word(self, word):
        pass
