import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import tensorflow as tf
import tensorflow_hub as hub

class SongLyricClassifier:

    def loadData(self):
        self.df = pd.read_csv("spotify_songs.csv", usecols = ['lyrics','playlist_genre'])

        genre_counts = self.df['playlist_genre'].value_counts()
        print(genre_counts)

    def preprocessText(self,lyrics):
        if pd.notna(lyrics):
            words = nltk.word_tokenize(lyrics.lower())
            words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(words)
        else:
            return ''

    def preprocessLyrics(self):
        self.df['lyrics'] = self.df['lyrics'].apply(self.preprocessText)

if __name__ == '__main__':
    songLyricClassifier = SongLyricClassifier()
    songLyricClassifier.loadData()
    songLyricClassifier.preprocessLyrics()
    print(songLyricClassifier.df.head(3))
    print("done")

