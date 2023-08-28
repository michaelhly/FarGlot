# FarGlot

A Transformer-based SocialNLP toolkit for [Farcaster](https://www.farcaster.xyz/).

## Installation

```
pip install farglot
```

## Examples

```python
from farglot import CastAnalyzer

sentiment_analyzer=CastAnalyzer.sequence_analzyer_from_model_name(
    hub_address="nemes.farcaster.xyz:2283",
    model_name="pysentimiento/robertuito-sentiment-analysis"
)

sentiment_analyzer.predict_cast(fid=2, hash_hex="0bcdcbf006ec22b79f37f2cf2a09c33413883937")
# {'NEG': 0.051998768001794815, 'NEU': 0.22470703721046448, 'POS': 0.7232941389083862}
sentiment_analyzer.predict_casts_by_fid(fid=2)
# {'NEG': 0.03734538331627846, 'NEU': 0.505352795124054, 'POS': 0.4573018550872803}
```

## Generate a Training Corpus from a [Hub](https://github.com/farcasterxyz/hub-monorepo/tree/main/apps/hubble)

### Install the FarGlot CLI

```
pip install "farglot[cli]"
```

### Define Training Set Columns

```json
{
  "name": "labels",
  "default_value": 1 // optional
}
```

For multi-label classfication:

```json
[
  {
    "name": "column_one",
    "default_value": 1 // optional
  },
  {
    "name": "column_two",
    "default_value": 2 // optional
  },
  {
    "name": "column_three",
    "default_value": 3 // optional
  }
]
```

### Usage

```sh
farglot init
farglot set-columns-path /path/to/column_configs.json
farglot set-hub-db-path /path/to/.rocks/rocks.hub._default
farglot new-training-set --out ./data/training-set.csv
```

### Tuning

TODO: Example of fine-tuning and uploading dataset and model to [Hugging Face](https://huggingface.co/)

### Tuning Resources

Not sure how to where to start? Check out the following blog posts on tuning an LLM:

- [Datasets and Preprocessing](https://michaelhly.com/posts/tune-llm-one)
- [Hyperparameters and Metrics](https://michaelhly.com/posts/tune-llm-two)

This largely is largely adapted off of [pysentimiento](https://github.com/pysentimiento/pysentimiento).
