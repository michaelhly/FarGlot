import numpy as np
from typing import Dict
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import f1_score, accuracy_score
from farglot.pretrained import default_auto_config, load_model_and_tokenizer

id2label = {0: "NEGATIVE", 1: "POSITIVE"}


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(label_ids, predictions),
        "f1": f1_score(label_ids, predictions),
    }


def binary_classifier(base_model: str) -> Trainer:
    model, tokenizer = load_model_and_tokenizer(
        base_model=base_model,
        auto_model_class=AutoModelForSequenceClassification,
        config=default_auto_config(base_model, id2label),
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=compute_metrics,
    )
