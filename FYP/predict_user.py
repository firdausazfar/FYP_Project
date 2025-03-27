import joblib
import numpy as np
from transformers import pipeline
from sklearn.preprocessing import StandardScaler

# === Load the trained model ===
model = joblib.load("models/depression_model.pkl")

# === Define label classes ===
label_names = ["Low", "Moderate", "High"]

# === Load sentiment analysis pipeline ===
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# === Input from user ===
tweet = input("Enter a tweet or statement: ")
spotify_values = input("Enter your Spotify features (comma-separated):\n"
                       "(BPM, Classical, Country, EDM, Folk, Gospel, Hip hop, Jazz, K pop, Latin, Lofi, Metal, Pop, R&B, Rap, Rock, Video game music)\n")

# === Sentiment prediction ===
sentiment = classifier(tweet)[0]['label']
sentiment_map = {
    "1 star": 0,
    "2 stars": 1,
    "3 stars": 2,
    "4 stars": 3,
    "5 stars": 4
}
sentiment_numeric = np.array([[sentiment_map.get(sentiment, 2)]])

# === Process Spotify input ===
spotify_features = np.array([float(v.strip()) for v in spotify_values.split(",")]).reshape(1, -1)

# === Normalize Spotify features (using default scaler) ===
scaler = StandardScaler()
scaled_spotify = scaler.fit_transform(spotify_features)  # NOTE: For demo — ideally reuse the training scaler

# === Combine features and predict ===
X_new = np.hstack((sentiment_numeric, scaled_spotify))
prediction = model.predict(X_new)[0]

print(f"\n🔮 Predicted Depression Level: {label_names[int(prediction)]}")