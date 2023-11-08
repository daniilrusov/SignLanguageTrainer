import pandas as pd
from collections import namedtuple


Word = namedtuple('Word', ['word', 'guide_path', 'category'])

class WordGenerator:
    def __init__(self, words_path):
        self.words_df = pd.read_csv(words_path) # df with cols: word, guide_path, category
    
    def get_random(self):
        sample = self.words_df.sample(1).iloc[0]
        return Word(sample.word, sample.guide_path, sample.category)

    def get_category(self, category):
        sample = self.words_df[self.words_df.category == category].sample(1).iloc[0]
        return Word(sample.word, sample.guide_path, sample.category)

    def get_word(self, word):
        sample = self.words_df[self.words_df.word == word].iloc[0]
        return Word(sample.word, sample.guide_path, sample.category)

    def get_words(self):
        words = [Word(sample.word, sample.guide_path, sample.category) for _, sample in self.words_df.iterrows()]
        return words

    def get_categories(self):
        categories = list(self.words_df.category.unique())
        return categories

