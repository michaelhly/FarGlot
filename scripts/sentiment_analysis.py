import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

OUT_DIR="data"

dataset = load_dataset(
    "csv",
    data_files={
        'test': 'data/test-set.csv',
        'train': "data/data-set.csv"
    })

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
   learning_rate=2e-5,
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
