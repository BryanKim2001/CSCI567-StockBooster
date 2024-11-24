import torch
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

text = "Can't believe my package from Amazon arrived damaged. Really disappointed."
inputs = tokenizer(text, return_tensors="pt")

print("Tokens :", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

print(predicted_class_id)

sentiment_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
sentiment = sentiment_labels[predicted_class_id]

print(f"Tweet: {text}")
print(f"Sentiment: {sentiment}")