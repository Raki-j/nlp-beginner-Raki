import torch
import torch.nn as nn
from tqdm import tqdm, trange # tqdm模块来显示任务进度条
from torch.optim import Adam
from tensorboardX import SummaryWriter
import pandas as pd
import os
from torchtext.legacy import data
from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import matplotlib.pyplot as plt
import numpy as np

from Model import RNN, CNN, LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_epochs = 5
batch_size = 512
learning_rate = 0.001
max_seq_length = 48
num_classes = 5
dropout_rate = 0.1
data_path = "data"
clip = 5

embed_size = 200
vectors = Vectors('glove.6B.200d.txt', 'C:/Users/Mechrevo/Desktop/AI/nlp-beginner/code-for-nlp-beginner-master/Task2-Text Classification (RNN&CNN)/embedding')
freeze = False

use_rnn = True
hidden_size = 256
num_layers = 1
bidirectional = True

use_lstm = False
num_filters = 200
kernel_sizes = [2, 3, 4]

def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None):
    TEXT = data.Field(lower=True, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(batch_first=True)
    train_fields = [(None, None), (None, None), ('text', TEXT), ('label', LABEL)]
    test_fields = [(None, None), (None, None), ('text', TEXT)]

    train_data = data.TabularDataset.splits(
        path=data_path,
        train='train.tsv',
        format='tsv',
        fields=train_fields,
        skip_header=True
    )[0]

    test_data = data.TabularDataset.splits(
        path='data',
        train='test.tsv',
        format='tsv',
        fields=test_fields,
        skip_header=True
    )[0]
    TEXT.build_vocab(train_data.text, vectors=vectors)
    LABEL.build_vocab(train_data.label)
    train_data, dev_data = train_data.split([0.8, 0.2])

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
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

if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT, LABEL = load_iters(batch_size, device, data_path, vectors)
    vocab_size = len(TEXT.vocab.itos)
    # build model
    if use_lstm:
        model = LSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate)
    elif use_rnn:
        model = RNN(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate)
    else:
        model = CNN(vocab_size, embed_size, num_classes, num_filters, kernel_sizes, dropout_rate)
    if vectors is not None:
        model.embed.from_pretrained(TEXT.vocab.vectors, freeze=freeze)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter('logs', comment="rnn")
    loss_history = []
    for epoch in trange(train_epochs, desc="Epoch"):
        model.train()
        ep_loss = 0
        for step, batch in enumerate(tqdm(train_iter, desc="Iteration")):
            (inputs, lens), labels = batch.text, batch.label
            outputs = model(inputs, lens)
            loss = loss_func(outputs, labels)
            ep_loss += loss.item()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if step % 10 == 0:
                loss_history.append(loss.item())
            writer.add_scalar('Train_Loss', loss, epoch)
            if step % 10 == 0:
                tqdm.write('Epoch {}, Step {}, Loss {}'.format(epoch, step, loss.item()))

        # evaluating
        model.eval()
        with torch.no_grad():
            corr_num = 0
            err_num = 0
            for batch in dev_iter:
                (inputs, lens), labels = batch.text, batch.label
                outputs = model(inputs, lens)
                corr_num += (outputs.argmax(1) == labels).sum().item()
                err_num += (outputs.argmax(1) != labels).sum().item()
            tqdm.write('Epoch {}, Accuracy {}'.format(epoch, corr_num / (corr_num + err_num)))
    if use_lstm:
        plt.title('LSTM Model')
    elif use_rnn:
        plt.title('RNN Model')
    else:
        plt.title('CNN Model')
    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.show()
    # predicting
    model.eval()
    with torch.no_grad():
        predicts = []
        for batch in test_iter:
            inputs, lens = batch.text
            outputs = model(inputs, lens)
            predicts.extend(outputs.argmax(1).cpu().numpy())
        test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')
        test_data["Sentiment"] = predicts
        test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('result.csv')
