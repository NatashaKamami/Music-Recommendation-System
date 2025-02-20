# Import libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models
model = joblib.load("music_recommender_model.joblib")
songs_data = joblib.load("songs_data.joblib")
combined_features = joblib.load("combined_features.joblib")

# Precompute lowercase song names and artist names for faster lookup
songs_data['name_lower'] = songs_data['name'].str.lower()
songs_data['artist_lower'] = songs_data['artist'].str.lower()

# Streamlit UI
st.title("Music Recommender 🎵🎶")
st.header('Find Songs Similar To Your Favorite Track')
st.write("Welcome to **RhythmIQ**! Get intelligent song recommendations based on deep learning and cosine similarity.")

# User input
query = st.text_input("Enter a song title or artist name:").strip().lower()

# Identify if input is a song or an artist
song_match = songs_data[songs_data['name_lower'] == query]
artist_match = songs_data[songs_data['artist_lower'] == query]

# Show search mode only if an artist is found
if not artist_match.empty:
    search_type = st.radio("Select Search Mode:", ["Show songs by the artist", "Recommend similar songs"])
else:
    search_type = "Recommend similar songs"  # Default for song searches

# Recommender function
def recommender(query, search_type, top_n=5):
    query = query.lower()

    # Check if query exists in data
    if query not in songs_data['name_lower'].values and query not in songs_data['artist_lower'].values:
        return f"'{query}' not found."

    # Case 1: User entered a song name
    if query in songs_data['name_lower'].values:
        song_idx = songs_data.index[songs_data['name_lower'] == query].tolist()[0]
        song_vector = combined_features[song_idx].reshape(1, -1)
        predicted_vector = model.predict(song_vector)
        similarities = cosine_similarity(predicted_vector, combined_features)
        similar_indices = similarities.argsort()[0][-top_n-1:-1]
        return songs_data.iloc[similar_indices][['name', 'artist', 'album', 'youtube_url']]

    # Case 2: User entered an artist name
    elif query in songs_data['artist_lower'].values:
        artist_songs = songs_data[songs_data['artist_lower'] == query][['name', 'artist', 'album', 'youtube_url']]

        # Show songs by the artist
        if search_type == "Show songs by the artist":
            return artist_songs.head(5)

        # Recommend songs similar to the artist style
        if search_type == "Recommend similar songs":
            artist_indices = artist_songs.index.tolist()
            artist_features = combined_features[artist_indices]
            artist_vector = artist_features.mean(axis=0).reshape(1, -1)
            predicted_vector = model.predict(artist_vector)

            similarities = cosine_similarity(predicted_vector, combined_features)
            similar_indices = similarities.argsort(axis=1)[:, -top_n-1:-1]

            recommended_songs = []
            for idx_list in similar_indices:
                recommended_songs.extend(songs_data.iloc[idx_list][['name', 'artist', 'album', 'youtube_url']].values)

            recommended = pd.DataFrame(recommended_songs, columns=['name', 'artist', 'album', 'youtube_url']).drop_duplicates().head(top_n)
            return recommended

# Display recommendations
if query:
    recommendations = recommender(query, search_type, top_n=5)
    if isinstance(recommendations, str):
        st.error(recommendations)
    elif not recommendations.empty:
        st.subheader(f"🎶 Top Recommendations for '{query}':")
        for _, row in recommendations.iterrows():
            st.write(f"- {row['name']} by {row['artist']} (Album: {row['album']})")
            st.markdown(f"[Listen on YouTube]({row['youtube_url']})")
    else:
        st.warning("No recommendations found.")
