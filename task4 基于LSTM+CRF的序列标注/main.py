import numpy as np
import torch.nn as nn
from utils import build_vocab
from utils import read_data
from utils import get_dataloader
from utils import pre_processing
from models import LSTM_CRF
import time
import torch
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors

n_classes = 5
batch_size = 250
embedding_size = 100
hidden_size = 20
epochs = 30

vectors = Vectors('glove.6B.100d.txt', 'C:/Users/Mechrevo/Desktop/nlp-beginner/code-for-nlp-beginner-master/Task2-Text Classification (RNN&CNN)/embedding')

def train(model, vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=None):
    model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    start_time = time.time()
    loss_history = []
    print("dataloader length: ", len(train_dataloader))
    model.train()
    for epoch in range(epochs):
        total_loss = 0.
        for idx, (inputs, targets, length_list) in enumerate(train_dataloader):

            model.zero_grad()
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (idx + 1) % 10 == 0 and idx:
                time_past = time.time() - start_time
                cur_loss = total_loss
                loss_history.append(cur_loss / (idx * batch_size))
                total_loss = 0
                print("epoch: ", epoch+1, "batch: ", round(idx * batch_size), "loss: ", cur_loss / (idx * batch_size))
    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('LSTM+CRF model')
    plt.show()

if __name__ == '__main__':
    x_train, y_train = read_data("data/conll2003/train.txt", 10000)
    x_test, y_test = read_data("data/conll2003/test.txt", 1000)
    pre_processing()
    word2idx, tag2idx, vocab_size = pre_processing()
    train_dataloader, max_length = get_dataloader(x_train, y_train, batch_size)
    test_dataloader, _ = get_dataloader(x_test, y_test, batch_size)
    train(LSTM_CRF, vocab_size, tag2idx, embedding_size, hidden_size, max_length=max_length, vectors=None)
