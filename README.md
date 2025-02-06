# Music Recommendation System: Travel-Themed Playlist Generator

## Problem Statement
With the growth of music streaming services, none provide country-specific playlists featuring artists from those specific countries. Travelers often struggle to find music that captures the vibe of different destinations and this project aims to fill that gap by generating location-based playlists using live Spotify data based on song similarity, popularity, and country-specific artists.

## Overview
The goal of this project is to create a travel-themed playlist generator, built upon the foundation of a general/basic music recommendation system in order to explore different models and techniques and pick out what would work best in creating the playlist generator.
The travel-themed playlist generator is a Streamlit-based web application that curates country-specific Spotify playlists depending on a user's travel destination. The system analyzes popular travel destinations and well-known artists from these countries, then selects songs with similar characteristics based on song names and popularity scores. The similarity of songs is calculated using cosine similarity, and a deep learning model with neural networks is used to enhance the recommendation quality.

## Data
Due to limitations when it came to scraping song data directly from Spotify using their web API, the data I decided to use is a combination of 2 kaggle datasets which had song data with all the features i needed for my basic music recommender system (song names, artist names, song audio features and song popularity scores.)
The resulting dataset contained 4993 entries and 18 columns which include: track_id, artist_name, track_name, popularity, duration_ms, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature and genre.

Links to the datasets: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset, https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks

## Methodology
#### 1. Exploratory Data Analysis
EDA helped in understanding summary statistics of the numeric features and to explore the relationships, patterns and trends that may exist in the dataset. 

#### 2. Handling Text Data
Since song names and artist names may contain important information, various Natural Language Processing (NLP) techniques were used.
I used NLTK for text tokenization and I also experimented with multiple text vectorization techniques, including TF-IDF, Word2Vec, Count Vectorization and GloVe to capture relationships and semantic similarities in the text data.

#### 3. Feature scaling
Most of the numeric features had values ranging between 0 and 1, minmax scaling was applied to normalize the numeric columns that contained data with different scales so that all the numeric data ranged from 0 to 1.

#### 4. Cosine Similarity Calculation
Cosine similarity is a measure of the similarity between two vectors, based on the cosine of the angle between them. It evaluates the orientation, not the magnitude, of the vectors. The similarity score ranges from 0 to 1, where 1 indicates identical vectors and 0 means no similarity.
To recommend songs, I computed cosine similarity based on combined features, which included a combination of the numeric features and text features.
After evaluation, Word2Vec provided the best similarity scores for song recommendation.
  
#### 5. Model Training for Recommendation
To improve recommendation quality, I incorporated the cosine similarity calculation into two models: K-Nearest Neighbors (KNN) Model and a Simple Deep Learning Model.
Both models were evaluated, and the deep learning model performed better since it had a lower MSE.

#### 6. Developing the Travel-Themed Playlist Generator
Once the best-performing text vectorization technique and model were identified, I expanded the system into a travel-themed playlist generator. The enhancements i made included:
- **Country-Specific Music Selection:** I created a dictionary of popular travel destinations and famous artists from those countries.

- **Live Data from Spotify API:** Instead of using a static dataset, I integrated Spotify's Web API to fetch real-time data on songs, artists, and popularity. However, due to API limitations, I couldn't access audio features (e.g., danceability, energy, acousticness), so the cosine similarity calculation was based on a combination of song name (text data) and popularity (numeric data). Then the deep learning model was used to make recommendations.

#### 7. Running the system on Streamlit
Streamlit allows users to input their travel destination and a playlist is generated for them. 

## Future Improvements
- Expanding the system to include more countries and artists.
- Integrating more audio features to enhance song selection.
- Incorporating user feedback to refine recommendations and allow for collaborative filtering.
