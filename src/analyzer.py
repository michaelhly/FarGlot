from typing_extensions import Self
from datasets import Dataset
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BatchEncoding,
    DataCollatorWithPadding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)
from torch import nn, tensor, softmax

from src.pretrained import load_model_and_tokenizer


class BaseAnalyzer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        tokenizer: PreTrainedTokenizerBase,
        training_args: TrainingArguments=TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=32
        )
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = training_args.per_device_eval_batch_size
        self.eval_trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        )

class AnalyzerForSequenceClassification(BaseAnalyzer):
    def __tokenize(self, batch: Union[str, List[str]]) -> BatchEncoding:
        return self.tokenizer(batch["text"], padding=False, truncation=True, max_length=self.tokenizer.model_max_length)

    @classmethod
    def from_model_name(cls, model_name: str, config: Optional[PretrainedConfig] = None) -> Self:
        model, tokenizer = load_model_and_tokenizer(
            base_model=model_name,
            auto_model_class=AutoModelForSequenceClassification,
            config=config
        )
        return cls(model, tokenizer)

    def predict(self, inputs: List[str]):
        # TODO(michael): Handle preprocessing
        data = { "text": [sentence for sentence in inputs] }
        dataset = Dataset.from_dict(data)
        dataset = dataset.map(self.__tokenize, batched=True, batch_size=self.batch_size)

        output = self.eval_trainer.predict(dataset)
        logits = tensor(output.predictions)
        probs = softmax(logits, dim=1).view(-1)
        id2label = self.model.config.id2label
        probas = { id2label[i]: probs[i].item() for i in id2label }
        # TODO(michael): Format probas

        return probas

class AnalyzerForTokenClassification(BaseAnalyzer):
    @classmethod
    def from_model_name(cls, model_name: str, config: Optional[PretrainedConfig] = None) -> Self:
        model, tokenizer = load_model_and_tokenizer(
            base_model=model_name,
            auto_model_class=AutoModelForTokenClassification,
            config=config
        )
        return cls(model, tokenizer)

    def predict(self, inputs: List[str]):
        raise NotImplementedError

