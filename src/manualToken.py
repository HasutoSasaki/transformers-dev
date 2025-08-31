import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import os

# Apple Silicon対応の環境変数を設定
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

sequence1 = "I've been waiting for a HuggingFace course my whole life."
sequence2 = "I hate this so much!"

# 1. 手動トークン化
tokens1 = tokenizer.tokenize(sequence1)
tokens2 = tokenizer.tokenize(sequence2)
print(f"Tokens 1: {tokens1}")
print(f"Tokens 2: {tokens2}")

# 2. トークンをIDに変換
ids1 = tokenizer.convert_tokens_to_ids(tokens1)
ids2 = tokenizer.convert_tokens_to_ids(tokens2)
print(f"Token IDs 1: {ids1}")
print(f"Token IDs 2: {ids2}")

# 3. [CLS]と[SEP]トークンを追加
input_ids1 = [tokenizer.cls_token_id] + ids1 + [tokenizer.sep_token_id]
input_ids2 = [tokenizer.cls_token_id] + ids2 + [tokenizer.sep_token_id]
print(f"Input IDs 1 with special tokens: {input_ids1}")
print(f"Input IDs 2 with special tokens: {input_ids2}")

# 4. バッチ処理（パディング + アテンションマスク）
max_len = max(len(input_ids1), len(input_ids2))

# パディング
padded_ids1 = input_ids1 + [tokenizer.pad_token_id] * (max_len - len(input_ids1))
padded_ids2 = input_ids2 + [tokenizer.pad_token_id] * (max_len - len(input_ids2))
print(f"Padded IDs 1: {padded_ids1}")
print(f"Padded IDs 2: {padded_ids2}")

# アテンションマスク
attention_mask1 = [1] * len(input_ids1) + [0] * (max_len - len(input_ids1))
attention_mask2 = [1] * len(input_ids2) + [0] * (max_len - len(input_ids2))
print(f"Attention mask 1: {attention_mask1}")
print(f"Attention mask 2: {attention_mask2}")

# バッチテンソル作成
batch_ids = torch.tensor([padded_ids1, padded_ids2])
batch_mask = torch.tensor([attention_mask1, attention_mask2])
print(f"Batch input IDs shape: {batch_ids.shape}")
print(f"Batch attention mask shape: {batch_mask.shape}")

# 5. 自動処理と比較（テンソル作成のみ）
print("Comparing with tokenizer's built-in method...")
auto_inputs = tokenizer([sequence1, sequence2], padding=True, return_tensors="pt")
print(f"Auto input IDs: {auto_inputs['input_ids']}")
print(f"Auto attention mask: {auto_inputs['attention_mask']}")

# テンソルが一致するか確認
print(f"Input IDs match: {torch.equal(batch_ids, auto_inputs['input_ids'])}")
print(f"Attention masks match: {torch.equal(batch_mask, auto_inputs['attention_mask'])}")