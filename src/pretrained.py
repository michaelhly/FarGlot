from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForPreTraining,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

def load_model_and_tokenizer(
    base_model: str,
    auto_model_class: AutoModelForPreTraining,
    max_length: int = 128,
    config: Optional[PretrainedConfig] = None,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = auto_model_class.from_pretrained(
        base_model,
        config=config,
        **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, config=config, **kwargs)
    tokenizer.model_max_length=max_length

    return model, tokenizer
