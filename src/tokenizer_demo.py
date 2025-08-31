from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

# token IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# decode
decoded = tokenizer.decode(ids)
print(decoded)
