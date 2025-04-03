# Music Recommendation System

## Overview
RhythmIQ is a music recommendation system that uses Natural Language Processing (NLP), deep learning, and cosine similarity to provide song recommendations based on user queries. The goal of the recommendation system is to enhance music discovery by incorporating deep learning, word embeddings, and numeric feature processing to improve song similarity detection. The model allows users to search for songs or artists and returns recommendations with clickable YouTube links for listening. It is implemented using Python and deployed with Streamlit for an interactive experience.

## How the system works
Unlike music streaming platforms where recommendations are based solely on basic metadata like genre and artist name, my recommendation system makes suggestions based on both textual and numeric features. 
When a user enters a song title or artist name, the system uses FastText embeddings to convert the text into numerical representations. This allows the model to capture meaningful relationships between words even across different languages. The numerical attributes are then normalized to ensure consistent scaling and combined with the text embeddings to form a comprehensive feature set for each song. 
A deep learning model is then trained to learn patterns in these features and cosine similarity is applied to make relevant song matches based on their charactersitics. 

## Data
The dataset was compiled using data from Spotify, YouTube and LastFM. Basic song information, such as song name, artist, album, and popularity, was collected from Spotify and stored in CSV files. To enhance the dataset with audio features, the corresponding songs were searched for on YouTube, downloaded, and analyzed using Librosa. This process allowed for the extraction of key audio metadata such as tempo, energy, spectral rolloff, chroma, and danceability. Lastly, data on the song genres, play count and number of listeners was collected from LastFm to further enrich the dataset. The resulting dataset contains 6,773 entries and 14 columns, which are:

- **song_id** – Unique identifier for the song on Spotify.
- **name** – Song title.
- **artist** – Name of the performing artist.
- **album** – Album in which the song belongs.
- **popularity** – Popularity score of the song on Spotify.
- **youtube_url** – Link to the song on YouTube.
- **tempo** – Speed of the song in beats per minute (BPM).
- **energy** – A measure of loudness and intensity over time. Higher values indicate louder and more energetic tracks and vice versa.
- **spectral_rolloff** – A measure of the frequency content of the song.
- **chroma** – A measure of intensity of different pitches/notes in the song. It helps analyze the harmonic and melodic content of a song. Higher values imply strong melodic structure and lower values suggest that a song relies more on beats and rhythm than melody.
- **danceability** – A score indicating how suitable the song is for dancing, derived from tempo and beats.
- **genre** - A list of genres that the song falls under.
- **listeners** - Number of unique listeners for the song.
- **plays** - Total number of times the song has been played.
  
## Methodology
### 1. Exploratory Data Analysis
EDA helped in understanding summary statistics of the numeric features and to explore the relationships, patterns and trends that may exist in the dataset.

### 2. Feature engineering
To analyze song replayability, a new feature called replayability was created by dividing the play count by the number of unique listeners to indicate how often a song is being replayed by its audience.
Higher replayability scores suggest that listeners frequently replay the song, indicating strong engagement and popularity, while lower scores imply that a song is played less frequently per listener, potentially reflecting lower listener engagement or appeal.

### 3. Multi-hot encoding
Since songs can belong to multiple genres, the genre column, which contained lists of genres, was transformed using multi-hot encoding. Each unique genre was converted into a separate binary column, where a value of 1 indicates that the song belongs to the genre, and 0 means it does not.
This transformation ensures that the dataset properly represents genre information in a format suitable for machine learning models.

### 4. Handling Text Data
Since song names can provide valuable insights into a song’s theme and vibe, various Natural Language Processing (NLP) techniques were applied.  
- **Text Tokenization:** I used NLTK to break down song names into individual tokens for further processing.  
- **Text Vectorization:** To capture relationships and semantic similarities in the text data, I experimented with multiple vectorization techniques, including:  
  - TF-IDF (Term Frequency-Inverse Document Frequency)  
  - Word2Vec 
  - Count Vectorization 
  - GloVe 
  - FastText 

After testing these approaches, I settled on FastText because of its ability to capture semantic relationships between words across different languages. This was particularly important since my dataset contained song names in multiple languages, not just English.  

### 5. Feature scaling
All the numeric columns in the dataset, contained data in different ranges. In order to ensure consistency across all the numerical features, Min-Max Scaling was applied to standardize the values within the range 0 to 1.

### 6. Cosine Similarity Calculation
Cosine similarity is a technique used to measure how similar two songs are by comparing their feature vectors. It checks how closely song features/charecteristics align by measuring the angle between their feature vectors. 
- **A smaller angle** means the songs are very similar.  
- **A larger angle** means they are less similar.
  
The similarity score ranges from 0 to 1, where 1 indicates identical song vectors and 0 means no similarity between the songs.

To generate music recommendations, I calculated cosine similarity using a combination of numeric features and text-based features (from song names, processed using FastText embeddings to convert them into numeric representations).  
This approach ensures that the recommendation system finds songs that are musically and thematically similar, helping users discover music that matches their preferences.

### 7. Model Training for Recommendation
To improve recommendation quality, I incorporated the cosine similarity calculation into two models: K-Nearest Neighbors (KNN) Model and a Simple Deep Learning Model. Both models were evaluated, and the deep learning model performed better since it had a lower Mean Squared Error (MSE).

### 8. Making Recommendations
Lastly, I created an end to end recommendation system that allows users to input a song name or an artist name in order to receive song recommendations. 
- **Song-Based Recommendations:**  
  - The model predicts the feature vector of the input song.  
  - Cosine similarity is computed against all songs in the dataset.  
  - The top 5 most similar songs are returned as recommendations.

 - **Artist-Based Recommendations:**  
    - **Retrieve songs by the artist:** The system filters and returns songs by the input artist.  
    - **Find similar songs from different artists:**  
       - The model first averages the feature vectors of all songs by the input artist to create an artist-level feature representation.  
       - This artist vector is then compared against all songs in the dataset using cosine similarity.  
       - The top 5 most similar songs, by different artists sharing similar musical characteristics, are returned.
         
This approach ensures personalized and diverse recommendations, helping users discover music that aligns with their preferences.  

### 9. Running the system on Streamlit
Streamlit allows users to input either a song name or an artist name and recommendations are given together with direct youtube links to the songs.
Users can access and use the app using this link: https://fr2wydfpgkzhe7a8ghespw.streamlit.app/ 

## Future Improvements
- Expanding the dataset to include more songs.
- Integrating Spotify API into the system to fetch real-time data.
- Incorporating user feedback to refine recommendations.
- Combine collaborative filtering with the existing content-based approach.
