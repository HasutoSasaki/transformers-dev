from transformers import pipeline

classifier = pipeline("zero-shot-classification")
results = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(f"======== output =======\n")
print(results)