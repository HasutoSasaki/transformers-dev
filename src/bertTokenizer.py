from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Using a Transformer network is simple")
print(inputs)

# save tokenizer
tokenizer.save_pretrained("directory_on_my_computer")
