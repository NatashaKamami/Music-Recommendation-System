# Importing required libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Load dataset
songs_data = pd.read_csv("music_data.csv").dropna()

# Preprocess numeric features
numeric_features = ['popularity', 'tempo', 'spectral_rolloff']
unscaled_numeric_features = ['danceability', 'energy', 'chroma']

# Scaling the unscaled numeric features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(songs_data[numeric_features])
unscaled_features = songs_data[unscaled_numeric_features].values

# Combine all numeric features
all_numeric_features = np.hstack([scaled_features, unscaled_features])

# Tokenizing text data
nltk.download('punkt')
def tokenize_text(text):
    return word_tokenize(str(text).lower())

songs_data['combined_tokens'] = songs_data[['name', 'album', 'artist']].apply(lambda x: sum(map(tokenize_text, x), []), axis=1)

# Train Word2Vec model
word2vec_model = Word2Vec(songs_data['combined_tokens'], vector_size=100, window=5, min_count=1, workers=4)

def get_song_vector(tokens):
    vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

songs_data['song_vector'] = songs_data['combined_tokens'].apply(get_song_vector)
song_vectors = np.array(songs_data['song_vector'].tolist())
combined_features = np.hstack([song_vectors, all_numeric_features])

# Train deep learning model
model = Sequential([
    Dense(256, activation='relu', input_shape=(combined_features.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(combined_features.shape[1], activation='linear')
])

model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(combined_features, combined_features, epochs=20, batch_size=32, 
          validation_split=0.2, callbacks=[early_stopping])

# Save models using joblib
joblib.dump(model, "music_recommender_model.joblib")
joblib.dump(songs_data, "songs_data.joblib")
joblib.dump(combined_features, "combined_features.joblib")

