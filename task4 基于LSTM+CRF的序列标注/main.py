import numpy as np
from utils import read_data
from utils import get_dataloader
from utils import pre_processing
from model import LSTM_CRF
import time
import torch
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors
from utils import compute_f1
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_classes = 5
batch_size = 250
embedding_size = 100
hidden_size = 20
epochs = 20
vectors = Vectors('glove.6B.100d.txt',
                  'C:/Users/Mechrevo/Desktop/nlp-beginner/code-for-nlp-beginner-master/Task2-Text Classification (RNN&CNN)/embedding')


def train(model, vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=None):
    model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    loss_history = []
    print("dataloader length: ", len(train_dataloader))
    model.train()
    f1_history = []
    idx2tag = {value: key for key, value in tag2idx.items()}
    for epoch in range(epochs):
        total_loss = 0.
        f1 = 0
        for idx, (inputs, targets, length_list) in enumerate(train_dataloader):

            model.zero_grad()
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            pred = model.predict(inputs, length_list)
            f1 += compute_f1(pred, targets, length_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if (idx + 1) % 10 == 0 and idx:
                cur_loss = total_loss
                loss_history.append(cur_loss / (idx+1))
                f1_history.append(f1 / (idx+1))
                total_loss = 0
                print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, idx*batch_size,
                                                                           cur_loss / (idx * batch_size), f1 / (idx+1)))

    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('LSTM+CRF model')
    plt.show()

    plt.plot(np.arange(len(f1_history)), np.array(f1_history))
    plt.title('train f1 scores')
    plt.show()

    model.eval()
    f1 = 0
    f1_history = []
    s = 0
    with torch.no_grad():
        for idx, (inputs, targets, length_list) in enumerate(test_dataloader):
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            pred = model.predict(inputs, length_list)
            f1 += compute_f1(pred, targets, length_list) * 32
    print("f1 score : {}, test size = {}".format(f1/3200, 3200))

if __name__ == '__main__':
    x_train, y_train = read_data("data/conll2003/train.txt", 14000)
    x_test, y_test = read_data("data/conll2003/test.txt", 3200)
    word2idx, tag2idx, vocab_size = pre_processing()
    train_dataloader, train_max_length = get_dataloader(x_train, y_train, batch_size)
    test_dataloader, test_max_length = get_dataloader(x_test, y_test, 32)
    train(LSTM_CRF, vocab_size, tag2idx, embedding_size, hidden_size, max_length=train_max_length, vectors=None)

