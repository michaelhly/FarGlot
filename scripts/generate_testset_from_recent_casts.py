import json
from datasets import Dataset
from transformers import pipeline

with open("data/recent_casts.json", "r") as f:
    casts = json.loads(f.readlines().pop())
    text_data = [cast["text"] for cast in casts]

    sentiment_classifier = pipeline("sentiment-analysis")
    sentiments = sentiment_classifier(text_data)

    predictions = { "hash": [], "text": [], "label": [], "score": [] }
    for cast, sentiment in zip(casts, sentiments):
        predictions["hash"].append(cast["hash"])
        predictions["text"].append(cast["text"])

        predictions["label"].append(sentiment["label"])
        score = 1.0 if sentiment["label"] == "POSITIVE" else -1.0
        predictions["score"].append(score)

    test_set = Dataset.from_dict(predictions)
    test_set.to_csv(path_or_buf="data/test-set.csv")
