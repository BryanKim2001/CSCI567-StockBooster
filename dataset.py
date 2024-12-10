import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data_dict, split, price_mean, price_std, tweet_dict):
        self.data = []
        self.price_mean = price_mean
        self.price_std = price_std

        for company, dates in data_dict[split].items():
            sorted_dates = sorted(dates.keys())
            first_price = 0.0

            for i in range(1, len(sorted_dates)):
                input_vector = [
                    (dates[sorted_dates[j]]["price"] - price_mean) / price_std if j >= 0  else first_price
                    for j in range(i - 3, i)
                ]
                tweet_available = 1 if sorted_dates[i] in tweet_dict.get(company, {}) else -1
                tweet_vector = [
                    tweet_dict.get(company, {}).get(               #appends tweet sentiment
                    sorted_dates[i], 2
                    ),
                    tweet_dict.get(company, {}).get(               #appends tweet sentiment
                    sorted_dates[i-1], 2
                    ),
                    tweet_dict.get(company, {}).get(               #appends tweet sentiment
                    sorted_dates[i-2], 2
                    )
                ]
                input_vector += tweet_vector
                label = 1 if dates[sorted_dates[i]]["price"] >= 0 else 0
                self.data.append({"input": input_vector, "label": label})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        return {
            "input": torch.tensor(sample["input"], dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.float32)
        }

