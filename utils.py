import pandas as pd
import os
import numpy as np

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
    return data_dict

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

def read_tweet_files(tweet_dir):
    