import joblib
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from twitter_api import get_recent_tweets
from spotify_api import get_recent_tracks, get_artist_genre_from_spotify

# === Load the trained model ===
model = joblib.load("models/depression_model.pkl")

# === Define label classes ===
label_names = ["Low", "Moderate", "High"]

# === Load sentiment analysis pipeline ===
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# === Sentiment mapping ===
sentiment_map = {
    "1 star": 0,
    "2 stars": 1,
    "3 stars": 2,
    "4 stars": 3,
    "5 stars": 4
}

# Fetch tweets and analyze sentiment
tweets = get_recent_tweets("FYPOus")  

print("\n=== Retrieved Tweets ===")
for t in tweets:
    print("-", t)

tweet_sentiments = [classifier(t[:512])[0]['label'] for t in tweets]
sentiment_scores = [sentiment_map.get(label, 2) for label in tweet_sentiments]
sentiment_avg = np.mean(sentiment_scores).reshape(1, -1)

print("\n=== Sentiment Analysis ===")
for label, score in zip(tweet_sentiments, sentiment_scores):
    print(f"{label} → {score}")
print(f"Average Sentiment Score: {sentiment_avg.flatten()[0]:.2f}")

# Fetch genres from recent Spotify tracks
tracks = get_recent_tracks()

print("\n=== Spotify Recent Tracks ===")
if not tracks:
    print("No tracks found. Please ensure your Spotify is active and connected.")
else:
    genre_list = []
    for track in tracks:
        genre = get_artist_genre_from_spotify(track['artist_id'])
        genre_list.append(genre)
        print(f"{track['name']} by {track['artist']} (Artist ID: {track['artist_id']}) → Genre: {genre}")

# Build genre frequency vector with partial string matching (case-insensitive)
all_genres = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "Latin", "Lofi", "K-pop", "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
genre_vector = []
for genre in all_genres:
    count = sum(1 for g in genre_list if genre.lower() in g.lower())
    genre_vector.append(count)

print("\n=== Spotify Genre Vector ===")
for genre, count in zip(all_genres, genre_vector):
    print(f"{genre}: {count}")

spotify_genres = np.array(genre_vector).reshape(1, -1)
# Dynamically extract genre column names for consistency with training
df = pd.read_csv("datasets/mxmh_survey_results.csv")
genre_columns = [col for col in df.columns if col.startswith("Frequency [")]

spotify_df = pd.DataFrame(spotify_genres, columns=genre_columns)

# === Normalize Spotify genre features using the trained scaler ===
scaler = joblib.load("models/spotify_scaler.pkl")
scaled_spotify = scaler.transform(spotify_df)

print("\n=== Scaled Spotify Features ===")
print(scaled_spotify)

# === Combine sentiment with normalized genre features and predict ===
sentiment_numeric = sentiment_avg
X_new = np.hstack((sentiment_numeric, scaled_spotify))

print("\n=== Combined Feature Vector ===")
print(X_new)

prediction = model.predict(X_new)[0]

print(f"\nPredicted Depression Level: {label_names[int(prediction)]}")