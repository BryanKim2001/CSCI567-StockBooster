import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

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

