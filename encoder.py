import tensorflow as tf
import numpy
import torch
import pandas as pd
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

root_folder = "stocknet-dataset/tweet/preprocessed"  
output_file = "combined_tweets.txt"  # Output text file to save train tweets
test_tweets = "test_tweets.txt"       # save test tweets
train_tweets_pd = []
test_tweets_pd = []
model.config.output_hidden_states = True
# Open the output file for writing
# edit this, a little messy now, clean up writing it to a text file
with open(output_file, "w") as outfile:
    # Walk through the folder structure
    for stock_ticker in os.listdir(root_folder):
        ticker_path = os.path.join(root_folder, stock_ticker)

        # Check if it's a directory (i.e., a stock ticker folder)
        #print(ticker_path)
        if os.path.isdir(ticker_path) and ticker_path == "stocknet-dataset/tweet/preprocessed/AAPL":
            print(len(os.listdir(ticker_path)))
            counter = 0
            for date_file in os.listdir(ticker_path):
                file_path = os.path.join(ticker_path, date_file)

                # Check if it's a file (not a subdirectory)
                if os.path.isfile(file_path) and counter <= int(.99*len(os.listdir(ticker_path))):
                    with open(file_path, "r") as infile:
                        # Read the tweets from the file
                        try:
                            tweets = [json.loads(line) for line in infile]
                            for tweet in tweets:
                                # Write each tweet to the output file, adding metadata
                                tweet_text = " ".join(tweet["text"])  # Combine tokenized text
                                metadata = f"Stock: {stock_ticker}, Date: {date_file}"
                                outfile.write(f"{metadata}\n{tweet_text}\n\n")
                                train_tweets_pd.append({"stock": stock_ticker, "date": date_file, "text": tweet_text})
                        except json.JSONDecodeError:
                            print(f"Skipping malformed file: {file_path}")
                elif os.path.isfile(file_path): #write 20% of tweets to the test file
                    with open(test_tweets, "w") as test_file:
                        with open(file_path, "r") as infile:
                            # Read the tweets from the file
                            try:
                                tweets = [json.loads(line) for line in infile]
                                for tweet in tweets:
                                    # Write each tweet to the output file, adding metadata
                                    tweet_text = " ".join(tweet["text"])  # Combine tokenized text
                                    #print("hi",tweet_text)
                                    metadata = f"Stock: {stock_ticker}, Date: {date_file}"
                                    test_file.write(f"{metadata}\n{tweet_text}\n\n")
                                    test_tweets_pd.append({"stock": stock_ticker, "date": date_file, "text": tweet_text})
                            except json.JSONDecodeError:
                                print(f"Skipping malformed file: {file_path}")
                counter+=1

def encode_tweets(text_list):
    # Tokenize the tweets
    #print(text_list)
    inputs = tokenizer(
        text_list,
        padding=True,           
        truncation=True,       
        max_length=128,        
        return_tensors="pt"     # Return PyTorch tensors
    )

    # Get embeddings
    with torch.no_grad():  # No gradient computation needed for inference
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # [batch_size, seq_length, hidden_dim]
        #last_hidden_states = hidden_states[-1]

    # Mean pooling to get sentence-level embeddings
    #print(hidden_states)
    embeddings = hidden_states[-1].mean(dim=1)  # Shape: [batch_size, hidden_dim]
    return embeddings

df_train = pd.DataFrame(train_tweets_pd)
df_test = pd.DataFrame(test_tweets_pd)

#print((df_test["text"].tolist()))

#for x in (df_test["text"].tolist()):
#    print(x)
embeddings = []
for e in df_test["text"].tolist():
    embeddings.append(encode_tweets(e))
#embeddings = encode_tweets(df_test["text"].tolist())
#print(df_test["text"])
#print(embeddings.shape())