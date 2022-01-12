import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128, dropout_rate=0.2, layer_num=2, max_seq_len=128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=layer_num, batch_first=True, dropout=dropout_rate,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def LSTM_leyer(self, x, hidden, lens):
        x = pack_padded_sequence(x, lens, batch_first=True)
        x, _ = self.lstm(x, hidden)
        x, _ = pad_packed_sequence(x)
        return torch.tensor(x)

    def init_weights(self):
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.layer_num, batch_size, self.hidden_size),
                weight.new_zeros(self.layer_num, batch_size, self.hidden_size))

    def forward(self, x, lens, hidden):
        x = self.embed(x)
        x = self.LSTM_leyer(x, hidden, lens)
        x = self.dropout(x)
        output = self.fc(x)
        return output
