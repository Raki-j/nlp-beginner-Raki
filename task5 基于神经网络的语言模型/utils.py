from torchtext.legacy import data
from torchtext.legacy.data import BucketIterator
import os

def read_data(path, max_length):
    with open(path, 'r', encoding="utf8") as f:
        poetries_list = []
        poetry = []
        for line in f:
            line = line.strip()
            if not line:
                if len(poetry) + len(line) <= max_length:
                    if poetry:
                        poetries_list.append(poetry)
                        poetry = []
            else:
                poetry.append(line)
        if poetry:
            poetries_list.append(poetry)
        return poetries_list

class PoetryDataset(data.Dataset):
    def __init__(self, text_field, path, max_length, **kwargs):
        fields = [("text", text_field)]
        raw_data = read_data(path, max_length)
        examples = []
        for text in raw_data:
            examples.append(data.Example.fromlist([text], fields))
        super(PoetryDataset, self).__init__(examples, fields, **kwargs)


def data_loader(eos_token="[EOS]", batch_size=32, device="cpu", data_path='data', max_length=128):
    TEXT = data.Field(eos_token=eos_token, batch_first=True, include_lengths=True)
    data_set = PoetryDataset(TEXT, os.path.join(data_path, "poetryFromTang.txt"), max_length)
    train_data, dev_data, test_data = data_set.split([0.8, 0.1, 0.1])

    TEXT.build_vocab(train_data)

    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )
    return train_iter, dev_iter, test_iter, TEXT
