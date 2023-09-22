import numpy as np
import pandas as pd
import random
import os

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub

class SongLyricClassifier:

    def loadData(self):
        self.songData = pd.read_csv("songs-small.csv", usecols=['lyrics', 'tag'])

        # genre_counts = self.songData['tag'].value_counts()
        # print(genre_counts)

    def tokenize(text):
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words

    def preprocessText(self, lyrics):
        if pd.notna(lyrics):
            # words = nltk.word_tokenize(lyrics.lower())
            words = self.tokenize(lyrics)
            words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(words)
        else:
            return ''
        

    def preprocessLyrics(self):
        self.songData['lyrics'] = self.songData['lyrics'].apply(self.preprocessText)

    def train(self):
        print("tdidf start")
        tfidfVectorizer = TfidfVectorizer(max_features=5000)
        tfidFeatures = tfidfVectorizer.fit_transform(self.songData['lyrics']).toarray()
        print("tdidf end")
        print("encoding start")
        label_encoder = LabelEncoder()
        self.songData['encoded_genre'] = label_encoder.fit_transform(self.songData['tag'])
        print("encoding end")
        print("training start")
        X = tfidFeatures  # Use the TF-IDF features here
        Y = self.songData['encoded_genre']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5000,)),  # Specify the input shape
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')  # Output layer with softmax for multiclass classification
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for multiclass classification
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test accuracy: {test_accuracy}')

def createNewCSV():
    original_df = pd.read_csv("songs.csv")
    num_rows_to_select = int(len(original_df) * 0.5)
    selected_rows = random.sample(range(len(original_df)), num_rows_to_select)
    new_df = original_df.iloc[selected_rows]
    new_df.to_csv("songs-small.csv", index=False)



if __name__ == '__main__':
    songLyricClassifier = SongLyricClassifier()
    songLyricClassifier.loadData()
    print("tokenizing start")
    songLyricClassifier.preprocessLyrics()
    print("tokenizing end")
    songLyricClassifier.train()
