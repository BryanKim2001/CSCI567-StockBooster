import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import math 

num_companies = 87

price_dict = {}
tweet_dict = {}
with open("price_dict.json", "r") as pf:
    price_dict = json.load(pf)
    del price_dict['GMRE']
with open("tweet_sentiment_dict.json", 'r') as tf:
    tweet_dict = json.load(tf)

def handle_missing_tweet_data(tweet_data, input_dim):
    if "sentiment" not in tweet_data:
        return {
            "sentiment": 2
        }
    return tweet_data

def construct_input_vector(price_dict, tweet_dict, company, date, input_dim):
    price_data = price_dict[company].get(date, {})
    adjusted_closing = price_data.get("price", 0)
    
    tweet_data = tweet_dict[company].get(date, 2)
    
    vector = [
        adjusted_closing,
        tweet_data
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
            price_change = price_dict[company][current_date]["price"]
            
            label = 1 if price_change > 0 else 0

            input_vectors.append(vector)
            labels.append(label)
    return np.array(input_vectors), np.array(labels)

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
input_dim = 2
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
    
    true_positive = ((predictions == 1) & (y_test_tensor == 1)).sum().item()
    true_negative = ((predictions == 0) & (y_test_tensor == 0)).sum().item()
    false_positive = ((predictions == 1) & (y_test_tensor == 0)).sum().item()
    false_negative = ((predictions == 0) & (y_test_tensor == 1)).sum().item()

    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    denominator = math.sqrt(
        (true_positive + false_positive) * (true_positive + false_negative) *
        (true_negative + false_positive) * (true_negative + false_negative)
    )
    mcc = numerator / denominator if denominator != 0 else 0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC Score: {mcc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
