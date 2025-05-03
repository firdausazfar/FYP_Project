import joblib
import numpy as np
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
tweet_sentiments = [classifier(t[:512])[0]['label'] for t in tweets]
sentiment_scores = [sentiment_map.get(label, 2) for label in tweet_sentiments]
sentiment_avg = np.mean(sentiment_scores).reshape(1, -1)

# Fetch genres from recent Spotify tracks
tracks = get_recent_tracks()
genre_list = [get_artist_genre_from_spotify(track['artist_id']) for track in tracks]

# Build genre frequency vector
all_genres = ["Classical", "Country", "EDM", "Folk", "Gospel", "Hip hop", "Jazz", "K pop", "Latin", "Lofi", "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music"]
genre_vector = [genre_list.count(g) for g in all_genres]
spotify_features = np.array(genre_vector).reshape(1, -1)

# === Normalize Spotify genre features using the trained scaler ===
scaler = joblib.load("models/spotify_scaler.pkl")
scaled_spotify = scaler.transform(spotify_features)

# === Combine sentiment with normalized genre features and predict ===
sentiment_numeric = sentiment_avg
X_new = np.hstack((sentiment_numeric, scaled_spotify))
prediction = model.predict(X_new)[0]

print(f"\n🔮 Predicted Depression Level: {label_names[int(prediction)]}")