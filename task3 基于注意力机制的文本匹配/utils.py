from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
import torch

def data_loader(batch_size=32, device="cuda", data_path='data', vectors=None):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = data.LabelField(batch_first=True)
    TREE = None

    fields = {'sentence1': ('premise', TEXT),
              'sentence2': ('hypothesis', TEXT),
              'gold_label': ('label', LABEL)}

    train_data, dev_data, test_data = data.TabularDataset.splits(
        path = data_path,
        train='snli_1.0_train.jsonl',
        validation='snli_1.0_dev.jsonl',
        test='snli_1.0_test.jsonl',
        format ='json',
        fields = fields,
        filter_pred=lambda ex: ex.label != '-'
    )
    TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(dev_data)
    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.premise) + len(x.hypothesis),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(
         test_data,
         batch_size=batch_size,
         device=device,
         sort=False,
         sort_within_batch=False,
         repeat=False,
         shuffle=False
    )

    return train_iter, dev_iter, test_iter, TEXT, LABEL