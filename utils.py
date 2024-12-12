import pandas as pd
import os
import numpy as np
import json
#from embeddings import get_tweet_sentiment

def create_data_dict(directory):
    data_dict = {}
    print(os.listdir(directory))
    for filename in os.listdir(directory):
        company_name = filename.split(".")[0]
        company_dict = {}
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        df = df.dropna()
        df['pct_change'] = df["Adj Close"].pct_change()
        df.loc[0, "pct_change"] = 0
        for _, row in df.iterrows():
            company_dict[row["Date"]] = {
                "price": row["pct_change"]
            }
        data_dict[company_name] = company_dict
    with open('price_dict.json', 'w') as file:
        json.dump(data_dict, file)
    return data_dict

def create_tweet_dict(tweet_directory):
    tweet_dict = {}
    for company in os.listdir(tweet_directory):
        company_path = os.path.join(tweet_directory, company)
        if (os.path.isdir(company_path)):
            tweet_dict[company] = {}
            for tweet_file in os.listdir(company_path):
                date = os.path.splitext(tweet_file)[0]
                file_path = os.path.join(company_path, tweet_file)

                with open(file_path, "r") as f:
                    tweet_set= set()
                    tweets = f.readlines()

                    if (tweets):
                        embeddings = []
                        #tweet_dict[company][date] = list(np.mean(embeddings, axis = 0))
                        for tweet in tweets:
                            text = json.loads(tweet)['text']
                            if (text not in tweet_set):
                                tweet_set.add(text)
                                embeddings.append(get_tweet_sentiment(preprocess_tweet(text.split(" "))))
                        tweet_dict[company][date] = np.mean(embeddings)
                    else:
                        tweet_dict[company][date] = 0
    print(len(tweet_dict.items()))
    with open('tweet_sentiment_dict.json', 'w') as file:
        json.dump(tweet_dict, file)
    return tweet_dict


def preprocess_tweet(tweet_tokens):
    modified_tweet_tokens = []
    for token in tweet_tokens:
        token = token.lower()
        token = token.replace("#", "")
        token = token.replace("\n", "")
        if (token.find("@") != -1):
            token = "AT_USER"
        elif (token.find("\\u") != -1):
            continue
        elif (token.find("http") != -1):
            token = "URL"
        elif (token.find("$") != -1):
            modified_tweet_tokens.append("$")
            token = token[1:]
        modified_tweet_tokens.append(token)
    return modified_tweet_tokens
        

def split_data(data_dict, split_ratio=0.8):
    train_data = {}
    test_data = {}
    for company, prices in data_dict.items():
        dates = sorted(prices.keys())
        split_index = int(len(dates) * split_ratio)

        train_dates = dates[:split_index]
        test_dates = dates[split_index:]

        train_data[company] = {date: prices[date] for date in train_dates}
        test_data[company] = {date: prices[date] for date in test_dates}

    return {"train": train_data, "test": test_data}

create_data_dict('data/price/raw')