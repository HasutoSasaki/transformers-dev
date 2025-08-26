from transformers import (AutoModelForSequenceClassification , AutoTokenizer)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)
print(outputs.logits.shape)
