from typing import Dict, Optional, Tuple
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from farglot.utils import gen_label2id


def default_auto_config(base_model: str, id2label: Dict[int, str]) -> AutoConfig:
    config: PretrainedConfig = AutoConfig.from_pretrained(base_model)
    config.id2label = id2label
    config.label2id = gen_label2id(id2label)
    return config


def load_model_and_tokenizer(
    base_model: str,
    auto_model_class: AutoModel,
    max_length: int = 128,
    config: Optional[AutoConfig] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = auto_model_class.from_pretrained(base_model, config=config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, config=config)
    tokenizer.model_max_length = max_length

    return model, tokenizer
