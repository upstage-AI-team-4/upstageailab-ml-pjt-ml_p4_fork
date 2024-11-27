# utils/twitter_collector.py
import tweepy
import pandas as pd
import os

class TwitterCollector:
    def __init__(self, api_key, api_secret, access_token, access_secret):
        self.auth = tweepy.OAuth1UserHandler(api_key, api_secret,
                                             access_token, access_secret)
        self.api = tweepy.API(self.auth)

    def collect_tweets(self, query, max_tweets, output_file):
        tweets = tweepy.Cursor(self.api.search_tweets,
                               q=query + ' -filter:retweets',
                               lang='ko',
                               tweet_mode='extended').items(max_tweets)

        data = [{'id': tweet.id_str,
                 'created_at': tweet.created_at,
                 'text': tweet.full_text} for tweet in tweets]

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"{len(df)}개의 트윗을 수집하여 {output_file}에 저장했습니다.")
