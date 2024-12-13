import torch
import numpy as np
import json
from dataset import StockDataset

from utils import create_data_dict, split_data, create_tweet_dict
from train import train_xgboost_model
from evaluate import evaluate_xgboost_model

from torch.utils.data import DataLoader
import logging
import json
logging.basicConfig(level=logging.INFO)

import os
print(os.getcwd())

data_dict = create_data_dict("data/price/raw")

tweet_dict = {}
with open('tweet_sentiment_dict.json', 'r') as tf:
    tweet_dict = json.load(tf)
print(len(tweet_dict.items()))
if (tweet_dict is None):
    tweet_dict = create_tweet_dict("data/raw_tweets/raw")

split_dict = split_data(data_dict, 0.8)
train_prices = []
for company, dates in split_dict["train"].items():
    for date, values in dates.items():
        price = values.get("price", None)
        if price is not None:
            train_prices.append(price)
price_mean = np.mean(train_prices)
price_std = np.std(train_prices)
print(price_mean)
print(price_std)

train_dataset = StockDataset(split_dict, split="train", price_mean=float(price_mean), price_std=float(price_std), tweet_dict=tweet_dict)
test_dataset = StockDataset(split_dict, split="test", price_mean=price_mean, price_std=price_std, tweet_dict=tweet_dict)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 6
model = train_xgboost_model(train_loader)

evaluate_xgboost_model(model, test_loader)

