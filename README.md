# Music Recommendation System

## Overview
RhythmIQ is a music recommendation system that uses Natural Language Processing (NLP), deep learning, and cosine similarity to provide song recommendations based on user queries. The goal of the recommendation system is to enhance music discovery by incorporating deep learning, word embeddings, and numeric feature processing to improve song similarity detection. The model allows users to search for songs or artists and returns recommendations with clickable YouTube links for listening. It is implemented using Python and deployed with Streamlit for an interactive experience.

## Data
The dataset was compiled using data from both Spotify and YouTube. Basic song information, such as song name, artist, album, and popularity, was collected from Spotify and stored in CSV files. To enhance the dataset with audio features, the corresponding songs were searched for on YouTube, downloaded, and analyzed using Librosa. This process allowed for the extraction of key audio metadata such as tempo, energy, spectral rolloff, chroma, and danceability. The final dataset contains 3,310 entries and 11 columns, which are:

- **song_id** – Unique identifier for the song on Spotify
- **name** – Song title
- **artist** – Name of the performing artist
- **album** – Album in which the song belongs
- **popularity** – Popularity score of the song on Spotify
- **youtube_url** – Link to the song on YouTube
- **tempo** – Speed of the song in beats per minute (BPM)
- **energy** – A measure of loudness and intensity over time ranging from 0 to 1. Higher values indicate louder and more energetic tracks and vice versa.
- **spectral_rolloff** – A measure of the frequency content of the song
- **chroma** – A measure intensity of different pitches/notes in the song, ranging from 0 to 1. It helps analyze the harmonic and melodic content of a song. Higher values imply strong melodic structure and lower values suggest that a song relies more on beats and rhythm than melody.
- **danceability** – A score from 0 to 1 indicating how suitable the song is for dancing, derived from tempo and beats. 

## Methodology
### 1. Exploratory Data Analysis
EDA helped in understanding summary statistics of the numeric features and to explore the relationships, patterns and trends that may exist in the dataset.

### 2. Handling Text Data
Since song names, artist names and album titles may contain important information, various Natural Language Processing (NLP) techniques were applied. I used NLTK for text tokenization and for text vectorization, I experimented with multiple text vectorization techniques, including TF-IDF, Word2Vec, Count Vectorization and GloVe to capture relationships and semantic similarities in the text data.

### 3. Feature scaling
Most of the numeric features had values ranging between 0 and 1, minmax scaling was applied to the numeric columns that contained different scaling so that all the numeric features ranged from 0 to 1.

### 4. Cosine Similarity Calculation
Cosine similarity is a measure of the similarity between two vectors, based on the cosine of the angle between them. It evaluates the orientation, not the magnitude, of the vectors. The similarity score ranges from 0 to 1, where 1 indicates identical vectors and 0 means no similarity. To recommend songs, I computed cosine similarity based on combined features, which included a combination of the numeric features and text features. After evaluation, Word2Vec provided the best similarity scores for song recommendation.

### 5. Model Training for Recommendation
To improve recommendation quality, I incorporated the cosine similarity calculation into two models: K-Nearest Neighbors (KNN) Model and a Simple Deep Learning Model. Both models were evaluated, and the deep learning model performed better since it had a lower Mean Squared Error (MSE).

### 6. Making Recommendations
Once the best-performing text vectorization technique and model were identified, I created an end to end recommendation system where when a user inputs a song name or an artist name, the model predicts its feature vector, and cosine similarity is computed against all songs. The top 5 most similar songs are returned as recommendations. For the artist based recommendation, the user can either retrieve songs by the artist or get recommendations for similar songs from different artists based on song features.

### 7. Running the system on Streamlit
Streamlit allows users to input either a song name or an artist name and recommendations are given together with direct youtube links to the songs.

## Future Improvements
- Expanding the dataset to include more songs.
- Integrating Spotify API into the system to fetch real-time data.
- Incorporating user feedback to refine recommendations.
- Combine collaborative filtering with the existing content-based approach.
