from twitter_pipeline import analyse_sentiment
from spotify_pipeline import load_spotify
from model_fusion import combine_features, train_model
import numpy as np
import pandas as pd

def encode_sentiments_to_numeric(sentiments):
    label_map = {
        "1 star": 0,
        "2 stars": 1,
        "3 stars": 2,
        "4 stars": 3,
        "5 stars": 4
    }

    return np.array([label_map.get(s, 2) for s in sentiments]).reshape(-1, 1)

if __name__ == "__main__":
    tweet_df = pd.read_csv("data/Combined_Data.csv")  
    tweets = tweet_df["tweet"].dropna().tolist()
    sentiments = analyse_sentiment(tweets)
    twitter_features = encode_sentiments_to_numeric(sentiments)

    spotify_features, labels = load_spotify("data/mxmh_survey_results.csv")

    min_length = min(len(twitter_features), len(spotify_features), len(labels))
    twitter_features = twitter_features[:min_length]
    spotify_features = spotify_features[:min_length]
    labels = labels[:min_length]

    X = combine_features(twitter_features, spotify_features)
    clf, acc = train_model(X, labels)

    print(f"✅ Model trained with {min_length} samples — Accuracy: {acc:.2f}")