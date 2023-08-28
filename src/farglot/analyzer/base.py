from abc import ABC, abstractmethod
from typing import List, Optional
from typing_extensions import Self
from transformers import PretrainedConfig


class BaseAnalyzer(ABC):
    @classmethod
    @abstractmethod
    def from_model_name(
        cls, model_name: str, config: Optional[PretrainedConfig] = None
    ) -> Self:
        raise NotImplementedError("from_model_name not implemented for BaseAnalyzer")

    @abstractmethod
    def predict(self, inputs: List[str]) -> bytes:
        raise NotImplementedError("predict not implemented for BaseAnalyzer")
