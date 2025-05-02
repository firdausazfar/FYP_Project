from transformers import pipeline
import pandas as pd

def load_tweets(csv_path):
    df = pd.read_csv(csv_path)
    return df['tweet'].tolist()

def analyse_sentiment(tweets):
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    tweets = [t[:512] for t in tweets]
    results = classifier(tweets, truncation=True)
    sentiments = [r['label'] for r in results]
    return sentiments

if __name__ == "__main__" :
    tweets = load_tweets("/datasets/Combined_Data.csv")
    sentiments = analyse_sentiment(tweets)
    for tweet, sent in zip(tweets, sentiments):
        print(f"{tweet[:50]}... => {sent}")
