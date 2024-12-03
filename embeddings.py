import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# tweet = ["great", "pennystock", "research", "$", "ahl", "$", "wmb", "$", "vrsn", "$", "goog", "i", "suggest", "URL"]
# text = " ".join(tweet)
# inputs = tokenizer(text, return_tensors="pt")

# print("Tokens :", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# with torch.no_grad():
#     logits = model(**inputs).logits

# print(logits)
# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]

# print(predicted_class_id)

# sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
# sentiment = sentiment_labels[predicted_class_id]

# print(f"Tweet: {text}")
# print(f"Sentiment: {sentiment}")

def get_tweet_sentiment(tweet):
    text = " ".join(tweet)
    inputs = tokenizer(text, return_tensors="pt")

    print(text)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]

    print(predicted_class_id, "\n")
    return predicted_class_id
# ----------------------------------------------------------------------------------------
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModel.from_pretrained("ProsusAI/finbert")

# def get_finbert_embedding(tweet):
#     text = " ".join(tweet)
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     print(text)
#     hidden_states = outputs.last_hidden_state
#     cls_embedding = hidden_states[:, 0, :]
#     return cls_embedding.squeeze(0).tolist()
    
# ----------------------------------------------------------------------------------------
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModel.from_pretrained("ProsusAI/finbert")

# def get_finbert_embedding(tweet_tokens):
#     text = " ".join(tweet_tokens)
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     hidden_states = outputs.hidden_states
#     print(outputs)
#     embeddings = hidden_states[-1].mean(dim=1)
#     print(embeddings)
#     return embeddings
#get_finbert_embedding(["iphone", "users", "are", "more", "intelligent", "than", "samsung", ",", "blackberry", "and", "htc", "owners", ",", "$", "aapl", "$", "bbry", ",", "URL"])
