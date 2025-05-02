import tweepy
from dotenv import load_dotenv
import os
import time

load_dotenv()


def get_recent_tweets(username, bearer_token=None, max_results=10):
    if bearer_token is None:
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    client = tweepy.Client(bearer_token=bearer_token)

    try:
        user = client.get_user(username=username)
        user_id = user.data.id

        tweets = client.get_users_tweets(id=user_id, max_results=min(max_results, 10), exclude=["retweets", "replies"])
        tweets_list = [tweet.text for tweet in tweets.data] if tweets.data else []
        return tweets_list
    
    except tweepy.TooManyRequests as e:
        reset_time = int(e.response.headers.get('x-rate-limit-reset'))
        wait_seconds = reset_time - int(time.time())
        print(f"Rate limit hit. Waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds + 1)
        return get_recent_tweets(username, bearer_token, max_results)


    
if __name__ == "__main__":
    BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    tweets = get_recent_tweets("FYPOus", BEARER_TOKEN)
    print(f"Retreived {len(tweets)} tweets.")
    for tweet in tweets:
        print("-", tweet)