import os
from datasets import load_dataset

from examples.classifiers.binary import binary_classifier

CHECKPOINTS_DIR = f"{os.getcwd()}/data/checkpoints"

dataset = load_dataset(
    "csv", data_files={"test": "data/test-set.csv", "train": "data/training-set.csv"}
)
trainer = binary_classifier("distilbert-base-uncased")


# PREPROCESS DATA
def preprocess_function(batch):
    tokenized_batch = trainer.tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
    )
    tokenized_batch["labels"] = [
        trainer.model.config.label2id[label] for label in batch["labels"]
    ]
    return tokenized_batch


trainer.train_dataset = dataset["train"].map(preprocess_function, batched=True)
trainer.eval_dataset = dataset["test"].map(preprocess_function, batched=True)

# SET TRAINING PARAMS

trainer.args.output_dir = CHECKPOINTS_DIR
trainer.args.eval_steps = 20
trainer.args.evaluation_strategy = "steps"
trainer.args.report_to = "wandb"
trainer.args.learning_rate = 2e-5
trainer.args.logging_steps = 5
trainer.args.per_device_train_batch_size = 16
trainer.args.per_device_eval_batch_size = 16
trainer.args.num_train_epochs = 2
trainer.args.weight_decay = 0.01
trainer.args.save_strategy = "epoch"

# TRAIN DATASETS

trainer.train()

# RUN EVALUATION AND RETURN METRICS

trainer.evaluate()
