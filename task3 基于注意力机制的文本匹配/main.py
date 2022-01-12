import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vectors
from models import ESIM
from tqdm import tqdm

from utils import data_loader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 36
hidden_size = 600 # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
epochs = 20
drop_out = 0.5
num_layers = 1
learning_rate = 4e-4
patience = 5
clip = 10
embedding_size = 300
device = 'cuda'
vectors = Vectors('glove.6B.300d.txt', 'C:/Users/Mechrevo/Desktop/nlp-beginner/code-for-nlp-beginner-master/Task2-Text Classification (RNN&CNN)/embedding')
data_path = 'data'


def train(train_iter, dev_iter, loss_func, optimizer, epochs, patience=5, clip=5):
    best_acc = -1
    patience_count = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for batch in tqdm(train_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label

            model.zero_grad()
            output = model(premise, premise_lens, hypothesis, hypothesis_lens).to(device)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            n += batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if n % 3600 == 0:
                print('epoch : {} step : {}, loss : {}'.format(epoch, int(n/3600), total_loss/n))
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    train_iter, dev_iter, test_iter, TEXT, LABEL = data_loader(batch_size, device, data_path, vectors)
    model = ESIM(num_features=(TEXT.vocab), hidden_size=hidden_size, embedding_size=embedding_size,
                 num_classes=4, vectors=TEXT.vocab.vectors, num_layers=num_layers,
                 batch_first=True, drop_out=0.5, freeze=False).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train(train_iter, dev_iter, loss_func, optimizer, epochs, patience, clip)
