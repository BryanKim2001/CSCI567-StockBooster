import torch
import numpy as np
from dataset import StockDataset
from utils import create_data_dict, split_data
from train import train_model
from evaluate import evaluate_model
from torch.utils.data import DataLoader

data_dict = create_data_dict("data/price/raw")
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

train_dataset = StockDataset(split_dict, split="train", price_mean=float(price_mean), price_std=float(price_std))
test_dataset = StockDataset(split_dict, split="test", price_mean=price_mean, price_std=price_std)

train_loader = DataLoader(train_dataset, batch_size=100000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 3
model = train_model(train_loader,input_size, device=device, weight_decay=1e-4)

evaluate_model(model, test_loader, device=device)