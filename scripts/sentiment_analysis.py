from datasets import load_dataset

data_set = load_dataset(
    "csv",
    data_files={
        'test': 'data/test-set.csv',
        'train': "data/data-set.csv"
    })
