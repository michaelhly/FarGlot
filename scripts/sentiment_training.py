import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

OUT_DIR = "data"

score2label = {1: "POSITIVE", -1: "NEGATIVE", 0: "NEUTRAL"}
label2score = {v: k for k, v in score2label.items()}

dataset = load_dataset(
    "csv", data_files={"test": "data/test-set.csv", "train": "data/training-set.csv"}
)

### PREPROCESSING ###

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(batch):
    tokenized_batch = tokenizer(
        batch["text"], padding=True, truncation=True, max_length=128
    )
    tokenized_batch["labels"] = [label2score[label] for label in batch["labels"]]
    return tokenized_batch


train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = train_dataset.map(preprocess_function, batched=True)

### TRAINING ###


def compute_metrics(eval_pred):
    load_accuracy = load("accuracy")
    load_f1 = load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_steps=20,
    evaluation_strategy="steps",
    report_to="wandb",
    learning_rate=2e-5,
    logging_steps=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
    compute_metrics=compute_metrics,
)


trainer.train()

### POST TRAINING ###

trainer.evaluate()
