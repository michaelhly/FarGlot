from typing_extensions import Self
from typing import List, Optional
from transformers import (
    AutoModelForTokenClassification,
    PretrainedConfig,
)

from farglot.analyzer.base import BaseAnalyzer
from farglot.pretrained import load_model_and_tokenizer


class AnalyzerForTokenClassification(BaseAnalyzer):
    @classmethod
    def from_model_name(
        cls, model_name: str, config: Optional[PretrainedConfig] = None
    ) -> Self:
        model, tokenizer = load_model_and_tokenizer(
            base_model=model_name,
            auto_model_class=AutoModelForTokenClassification,
            config=config,
        )
        return cls(model, tokenizer)

    def predict(self, inputs: List[str]):
        raise NotImplementedError(
            "predict not implemented for AnalyzerForTokenClassification"
        )
