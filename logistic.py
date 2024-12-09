import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

def create_dataset(price_dict, tweet_dict, window=3):
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

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
classes = np.unique(y_train)
class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))
print("Class Weights: ", class_weights_dict)

logistic_model = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights_dict)
print("\nTraining Logistic Regression Model...")
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report: ")
print(classification_report(y_test, y_pred, target_names=["Down", 'Up']))
