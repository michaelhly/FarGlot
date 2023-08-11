from datasets import load_dataset, Features,Value

context_feat = Features({'text': Value(dtype='string', id=None)})
test_set = load_dataset("csv", data_files=['data/test-set.csv'])
print(test_set["train"][0])