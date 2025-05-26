from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = input("Enter a sentence for sentiment analysis: ")

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

probs = F.softmax(logits, dim=1)

pred_class = torch.argmax(probs, dim=1).item()
labels = ["Negative", "Positive"]
pred_label = labels[pred_class]

print(f"\nInput Text: {text}")
print(f"Predicted Sentiment: {pred_label}")
print(f"Probabilities: {probs.squeeze().tolist()}")