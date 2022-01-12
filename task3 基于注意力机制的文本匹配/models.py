import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]

        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens.cpu(), batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]
        return recover_output

class Input_Encoding(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, vectors,
                 num_layers=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.num_features = num_features
        self.num_hidden = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop_out)
        self.embedding = nn.Embedding.from_pretrained(vectors).cuda()
        self.bilstm = BiLSTM(embedding_size, hidden_size, drop_out, num_layers)

    def forward(self, x, lens):
        #x = torch.LongTensor(x)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.bilstm(x, lens)
        return x

class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).to(device)
        self.softmax_2 = nn.Softmax(dim=2).to(device)

    def forward(self, a_, b_):
        e = torch.matmul(a_, b_.transpose(1, 2)).to(device)

        a_tilde = (self.softmax_2(e)).bmm(b_)
        b_tilde = (self.softmax_1(e).transpose(1, 2)).bmm(a_)

        m_a = torch.cat([a_, a_tilde, a_ - a_tilde, a_ * a_tilde], dim=-1)
        m_b = torch.cat([b_, b_tilde, b_ - b_tilde, b_ * b_tilde], dim=-1)

        return m_a, m_b

class Inference_Composition(nn.Module):
    def __init__(self, num_features, m_size, hidden_size, num_layers, embedding_size, batch_first=True,drop_out=0.5):
        super(Inference_Composition,self).__init__()
        self.linear = nn.Linear(4 * hidden_size, hidden_size).to(device)
        self.bilstm = BiLSTM(hidden_size, hidden_size, drop_out, num_layers).to(device)
        self.drop_out = nn.Dropout(drop_out).to(device)

    def forward(self, x, lens):
        x = self.linear(x)
        x = self.drop_out(x)
        x = self.bilstm(x, lens)

        return x

class Prediction(nn.Module):
    def __init__(self, v_size, mid_size, num_classes=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(v_size, mid_size), nn.Tanh(),
            nn.Linear(mid_size, num_classes)
        ).to(device)

    def forward(self, a, b):
        v_a_avg = F.avg_pool1d(a.transpose(1, 2), a.size(1)).squeeze(-1)
        v_a_max = F.max_pool1d(a.transpose(1, 2), a.size(1)).squeeze(-1)

        v_b_avg = F.avg_pool1d(b.transpose(1, 2), b.size(1)).squeeze(-1)
        v_b_max = F.max_pool1d(b.transpose(1, 2), b.size(1)).squeeze(-1)


        out_put = torch.cat((v_a_avg, v_a_max, v_b_avg, v_b_max), dim=-1)

        return self.mlp(out_put)

class ESIM(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, num_classes=4, vectors=None,
                 num_layers=1, batch_first=True, drop_out=0.5, freeze=False):
        super(ESIM, self).__init__()
        self.embedding_size = embedding_size
        self.input_encoding = Input_Encoding(num_features, hidden_size, embedding_size, vectors,
                 num_layers=1, batch_first=True, drop_out=0.5)
        self.local_inference = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(num_features, 4 * hidden_size, hidden_size,
                                                           num_layers, embedding_size=embedding_size,
                                                           batch_first=True, drop_out=0.5)
        self.prediction = Prediction(4 * hidden_size, hidden_size, num_classes, drop_out)

    def forward(self, a, len_a, b, len_b):
        a_bar = self.input_encoding(a, len_a)
        b_bar = self.input_encoding(b, len_b)

        m_a, m_b = self.local_inference(a_bar, b_bar)

        v_a = self.inference_composition(m_a, len_a)
        v_b = self.inference_composition(m_b, len_b)

        out_put = self.prediction(v_a, v_b)

        return out_put


