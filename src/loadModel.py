from transformers import BertModel
# AutoModelを使うこともできますが、チェックポイントに依存しないコードを生成するために、BertModelを直接使用します
model = BertModel.from_pretrained("bert-base-cased")

# モデルを保存する例
model.save_pretrained("directory_on_my_computer")

print(model)    