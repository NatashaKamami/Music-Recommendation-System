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
from gensim.models import FastText

# Load dataset
songs_data = pd.read_csv("new_music_data.csv").dropna()

# Preprocessing numeric features
numeric_features = [ 'popularity', 'tempo', 'energy', 'spectral_rolloff', 'chroma', 'danceability',
       'listeners', 'plays', 'replayability', 'genre_afrobeats',
       'genre_afrofusion', 'genre_afrohouse', 'genre_afropop',
       'genre_afroswing', 'genre_alternative', 'genre_amapiano', 'genre_blues',
       'genre_bongo', 'genre_classic', 'genre_country', 'genre_dance',
       'genre_dancehall', 'genre_disco', 'genre_drill', 'genre_edm',
       'genre_electronic', 'genre_folk', 'genre_funk', 'genre_genge',
       'genre_gengetone', 'genre_gqom', 'genre_grime', 'genre_grunge',
       'genre_hiphop', 'genre_house', 'genre_indie', 'genre_jazz',
       'genre_kenyan', 'genre_kenyan drill', 'genre_kenyan hiphop',
       'genre_kenyan rnb', 'genre_latin', 'genre_metal', 'genre_mugithi',
       'genre_neosoul', 'genre_pop', 'genre_punk', 'genre_rap', 'genre_reggae',
       'genre_reggaeton', 'genre_rnb', 'genre_rock', 'genre_singer-songwriter',
       'genre_soft rock', 'genre_soul', 'genre_synthpop',
       'genre_tanzanian hiphop', 'genre_trap', 'genre_uk rap']

# Scale the numeric features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(songs_data[numeric_features])

# Tokenizing text data
nltk.download('punkt')
def tokenize_text(text):
    return word_tokenize(str(text).lower())

songs_data['name_tokens'] = songs_data['name'].apply(tokenize_text)


# Train FastText model on the tokenized song names
fasttext_model = FastText(sentences=songs_data['name_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Get vector representation for each song by averaging token vectors
def get_song_vector(tokens):
    vectors = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)  # Return zero vector if no valid tokens

# Apply to each song
songs_data['song_vector'] = songs_data['name_tokens'].apply(get_song_vector)
song_vectors = np.array(songs_data['song_vector'].tolist())

# Stack text-based and numeric features together
combined_features = np.hstack([song_vectors, scaled_features])
combined_features = np.asarray(combined_features, dtype=np.float32)


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

