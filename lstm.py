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

def handle_missing_tweet_data(tweet_data):
    if "sentiment" not in tweet_data:
        return {
            "sentiment": 2
        }
    return tweet_data

def add_advanced_features(price_dict):
    for company, dates in price_dict.items():
        sorted_dates = sorted(dates.keys())
        for i in range(len(sorted_dates)):
            date = sorted_dates[i]

            if i >= 4:  
                ma5 = np.mean([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 4, i + 1)])
            else:
                ma5 = 0

            if i >= 9:  
                ma10 = np.mean([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 9, i + 1)])
            else:
                ma10 = 0

            if i >= 4:
                volatility = np.std([price_dict[company][sorted_dates[j]]["adjusted_closing"] for j in range(i - 4, i + 1)])
            else:
                volatility = 0

            if i >= 3:
                momentum = price_dict[company][date]["adjusted_closing"] - price_dict[company][sorted_dates[i - 3]]["adjusted_closing"]
            else:
                momentum = 0

            price_dict[company][date]["ma5"] = ma5
            price_dict[company][date]["ma10"] = ma10
            price_dict[company][date]["volatility"] = volatility
            price_dict[company][date]["momentum"] = momentum

    return price_dict

def construct_input_vector(price_dict, tweet_dict, company, date):
    price_data = price_dict[company].get(date, {})
    adjusted_closing = price_data.get("adjusted_closing", 0)
    high = price_data.get('high', 0)
    low = price_data.get('low', 0)
    ma5 = price_data.get("ma5", 0)
    ma10 = price_data.get("ma10", 0)
    volatility = price_data.get("volatility", 0)
    momentum = price_data.get("momentum", 0)
    
    tweet_data = tweet_dict[company].get(date, {})
    tweet_data = handle_missing_tweet_data(tweet_data)
    vector = [
        adjusted_closing, high, low, ma5, ma10, volatility, momentum,
        tweet_data['sentiment']
    ]
    return vector

def create_dataset(price_dict, tweet_dict, window=5):
    input_vectors = []
    labels = []
    for company in price_dict.keys():
        dates = sorted(price_dict[company].keys())
        for i in range(window, len(dates)):
            vector = []
            for j in range(i - window, i):
                date = dates[j]
                vector.extend(construct_input_vector(price_dict, tweet_dict, company, date))
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

class LSTMModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output

X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
X_tensor = X_tensor.view(-1, 5, 8)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


feature_dim = 8
hidden_dim = 64
num_layers = 1
output_dim = 2
learning_rate = 0.001
num_epochs = 100
batch_size = 64

model = LSTMModel(feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
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
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, axis = 1)
    accuracy = (predictions == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy: .4f}")
