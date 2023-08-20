import json
from datasets import Dataset
from transformers import pipeline

with open("data/recent-casts.json", "r") as f:
    casts = json.loads(f.readlines().pop())
    text_data = [cast["text"] for cast in casts]

    sentiment_classifier = pipeline("sentiment-analysis")
    sentiments = sentiment_classifier(text_data)

    predictions = {"hash": [], "text": [], "labels": []}
    for cast, sentiment in zip(casts, sentiments):
        if not cast["text"]:
            continue

        predictions["hash"].append(cast["hash"])
        predictions["text"].append(cast["text"])
        predictions["labels"].append(sentiment["label"])

    test_set = Dataset.from_dict(predictions)
    test_set.to_csv(path_or_buf="data/test-set.csv")
