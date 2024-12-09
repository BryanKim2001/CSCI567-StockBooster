import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

num_companies = 87

price_dict = {}
tweet_dict = {}
with open("price_dict.json", "r") as pf:
    price_dict = json.load(pf)
    del price_dict['GMRE']
with open("super_tweet_dict.json", 'r') as tf:
    tweet_dict = json.load(tf)

embeddingLengths = set()

def handle_missing_tweet_data(tweet_data, input_dim):
    if "sentiment" not in tweet_data or "embedding" not in tweet_data:
        return {
            "sentiment": 2,
            "embedding": [0.0] * input_dim
        }
    return tweet_data

def add_advanced_features(price_dict):
    for company, dates in price_dict.items():
        sorted_dates = sorted(dates.keys())
        for i in range(len(sorted_dates)):
            date = sorted_dates[i]
            # 5-day moving average
            if i >= 4:  # Need at least 5 days for the MA
                ma5 = np.mean([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 4, i + 1)])
            else:
                ma5 = 0

            # 10-day moving average
            if i >= 9:  # Need at least 10 days for the MA
                ma10 = np.mean([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 9, i + 1)])
            else:
                ma10 = 0

            # 5-day volatility (standard deviation)
            if i >= 4:
                volatility = np.std([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 4, i + 1)])
            else:
                volatility = 0

            # 3-day momentum
            if i >= 3:
                momentum = price_dict[company][date]["adjusted_closing"] - price_dict[company][sorted_dates[i - 3]]["adjusted_closing"]
            else:
                momentum = 0

            # Add features to the current date
            price_dict[company][date]["ma5"] = ma5
            price_dict[company][date]["ma10"] = ma10
            price_dict[company][date]["volatility"] = volatility
            price_dict[company][date]["momentum"] = momentum

    return price_dict

def construct_input_vector(price_dict, tweet_dict, company, date, input_dim):
    price_data = price_dict[company].get(date, {})
    adjusted_closing = price_data.get("adjusted_closing", 0)
    high = price_data.get('high', 0)
    low = price_data.get('low', 0)
    ma5 = price_data.get("ma5", 0)
    ma10 = price_data.get("ma10", 0)
    volatility = price_data.get("volatility", 0)
    momentum = price_data.get("momentum", 0)
    
    tweet_data = tweet_dict[company].get(date, {})
    tweet_data = handle_missing_tweet_data(tweet_data, input_dim)
    embeddingLengths.add(len(tweet_data['embedding']))
    vector = [
        adjusted_closing, high, low, ma5, ma10, volatility, momentum,
        tweet_data['sentiment'],
        *tweet_data["embedding"]
    ]
    return vector

def create_dataset(price_dict, tweet_dict, window=3, input_dim=768):
    input_vectors = []
    labels = []
    for company in price_dict.keys():
        dates = sorted(price_dict[company].keys())
        for i in range(window, len(dates)):
            vector = []
            for j in range(i - window, i):
                date = dates[j]
                vector.extend(construct_input_vector(price_dict, tweet_dict, company, date, input_dim))
            current_date = dates[i]
            previous_date = dates[i-1]
            price_change = (
                price_dict[company][current_date]["adjusted_closing"]
                - price_dict[company][previous_date]["adjusted_closing"]
            )
            label = 1 if price_change > 0 else 0

            input_vectors.append(vector)
            labels.append(label)
    return np.array(input_vectors), np.array(labels)

price_dict = add_advanced_features(price_dict)
X, y = create_dataset(price_dict, tweet_dict)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size = 0.2, random_state = 42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim, seq_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim * seq_len, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        flattened = transformer_out.view(transformer_out.size(0), -1)
        output = self.fc(flattened)
        return output

# Hyperparameters
input_dim = 776
seq_len = 3
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 1
output_dim = 2
learning_rate = 0.01
num_epochs = 50
batch_size = 5000


X_train_tensor = X_train_tensor.view(-1, seq_len, input_dim)
X_test_tensor = X_test_tensor.view(-1, seq_len, input_dim)

model = TransformerModel(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                         hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, seq_len=seq_len)
unique, counts = np.unique(y_train, return_counts=True)
class_counts = torch.tensor(counts, dtype=torch.float32)
class_weights = 1.0 / class_counts
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    correct = 0
    total = 0
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, axis=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
        train_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item(): .4f}, Training Accuracy: {train_accuracy:.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predictions = torch.argmax(test_outputs, axis = 1)
    accuracy = (predictions == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {accuracy: .4f}")
