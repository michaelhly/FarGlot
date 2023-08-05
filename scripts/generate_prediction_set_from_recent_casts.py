import json
from datasets import Dataset
from transformers import pipeline

with open("data/recent_casts.json", "r") as f:
    raw_data = json.loads(f.readlines()[0])
    casts = raw_data["result"]["casts"]
    text_data = [cast["text"] for cast in casts]

    sentiment_classifier = pipeline("sentiment-analysis")
    sentiments = sentiment_classifier(text_data)

    predictions = {"hash": [], "text": [], "label": [], "score": [] }
    for cast, sentiment in zip(casts, sentiments):
        predictions["hash"].append(cast["hash"])
        predictions["text"].append(cast["text"])

        predictions["label"].append(sentiment["label"])
        score = 1.0 if sentiment["label"] == "POSITIVE" else 0.0
        predictions["score"].append(score)

    pred_set = Dataset.from_dict(predictions)
    pred_set.set_format(type="numpy")
    pred_set.export(filename="data/pretrained-prediction-set.tfrecord")
