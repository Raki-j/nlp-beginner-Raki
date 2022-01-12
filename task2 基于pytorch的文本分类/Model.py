import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes,
                 bidirectional=True, dropout_rate=0.3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        if not bidirectional:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.init()

    def init(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, lens):
        embeddings = self.embed(x)
        output, _ = self.rnn(embeddings)

        real_output = output[range(len(lens)), lens - 1]
        out = self.fc(self.dropout(real_output))
        return out
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes,
                 bidirectional=True, dropout_rate=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        if not bidirectional:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.init()

    def init(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, lens):
        embeddings = self.embed(x)
        output, _ = self.rnn(embeddings)

        real_output = output[range(len(lens)), lens - 1]
        out = self.fc(self.dropout(real_output))
        return out
class CNN(nn.Module):
    def __int__(self, vocab_size, embed_size, num_classes, num_filters=100, kernel_size=[2, 3, 4], dropout_rate=0.3):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_size), padding=(k - 1, 0))
            for k in kernel_size
        ])
        self.fc = nn.Linear(len(kernel_size) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x).squeeze(3))
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x, lens):
        embed = self.embed(x).unsqueeze(1)

        conv_results = [self.conv_and_pool(embed, conv) for conv in self.convs]

        out = torch.cat(conv_results, 1)
        return self.fc(self.dropout(out))