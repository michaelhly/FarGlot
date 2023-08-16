import os
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

OUT_DIR="data"

dataset = load_dataset(
    "csv",
    data_files={
        'test': 'data/test-set.csv',
        'train': "data/training-set.csv"
    })

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "sentiment-training"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"
# wandb base url
os.environ["WANDB_BASE_URL"] = "http://localhost:8080"

### PREPROCESSING ###

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = train_dataset.map(preprocess_function, batched=True)

### TRAINING ###

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
training_args = TrainingArguments(
   output_dir=OUT_DIR,
   report_to="wandb",
   learning_rate=2e-5,
   logging_steps=5,
   per_device_train_batch_size=32,
   per_device_eval_batch_size=32,
   evaluation_strategy="steps",
   eval_steps=20,
   max_steps=100,
   save_steps=100,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
   compute_metrics=compute_metrics,
)


trainer.train()

### POST TRAINING ###

trainer.evaluate()
