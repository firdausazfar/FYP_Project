from twitter_pipeline import analyse_sentiment
from spotify_pipeline import load_spotify
from model_fusion import combine_features, train_model
import numpy as np
import pandas as pd
import time

def encode_sentiments_to_numeric(sentiments):
    label_map = {
        "1 star": 0,
        "2 stars": 1,
        "3 stars": 2,
        "4 stars": 3,
        "5 stars": 4
    }

    return np.array([label_map.get(s, 2) for s in sentiments]).reshape(-1, 1)

def simplify_labels(y):
    return np.array([
        0 if score <= 2 else     # low/none
        1 if score <= 6 else     # moderate
        2                        # high/severe
        for score in y
    ])

if __name__ == "__main__":
    tweet_df = pd.read_csv("datasets/Combined_Data.csv")  
    print("Available columns in Combined_Data.csv:", tweet_df.columns.tolist())

    tweet_df = tweet_df.dropna(subset=["statement"]) 
    print(f"Loaded tweet_df shape: {tweet_df.shape}")
    tweets = tweet_df["statement"].dropna().tolist()
    print(f"Loaded {len(tweets)} tweets")

    print("Starting sentiment analysis...")
    sentiments = []
    batch_size = 1000
    total_batches = (len(tweets) + batch_size - 1) // batch_size
    for i in range(total_batches):
        batch = tweets[i * batch_size:(i + 1) * batch_size]
        print(f"Processing batch {i + 1}/{total_batches}...")
        start_time = time.time()
        batch_sentiments = analyse_sentiment(batch)
        elapsed = time.time() - start_time
        print(f"Batch {i + 1} done in {elapsed:.2f} seconds")
        sentiments.extend(batch_sentiments)
    print(f"Sentiments length: {len(sentiments)}")
    twitter_features = encode_sentiments_to_numeric(sentiments)
    print(f"Encoded Twitter features shape: {twitter_features.shape}")

    print("Loading Spotify genre data...")
    spotify_features, labels = load_spotify("datasets/mxmh_survey_results.csv")
    print(f"Spotify features shape: {spotify_features.shape}, Labels shape: {labels.shape}")

    min_length = min(len(twitter_features), len(spotify_features), len(labels))
    print(f"Truncating all features and labels to min_length: {min_length}")
    twitter_features = twitter_features[:min_length]
    spotify_features = spotify_features[:min_length]
    labels = labels[:min_length]

    labels = simplify_labels(labels)
    print(f"Simplified labels shape: {labels.shape}")

    X = combine_features(twitter_features, spotify_features)
    print(f"Combined feature matrix shape: {X.shape}")
    
    from model_fusion import simulate_fused_training
    clf, acc = simulate_fused_training(twitter_features, labels, spotify_features, labels)
    print(f"Model trained with {min_length} samples — Accuracy: {acc:.2f}")