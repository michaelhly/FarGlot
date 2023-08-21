from typing import Dict, List, Union


def gen_label2id(id2label: Union[Dict[int, str], List[str]]) -> Dict[str, int]:
    if type(id2label) is list:
        return {str(i): label for i, label in enumerate(id2label)}
    else:
        return {v: k for k, v in id2label.items()}
