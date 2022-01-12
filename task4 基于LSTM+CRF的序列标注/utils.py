import torch
from torch.utils.data import DataLoader, Dataset

def read_data(path, length):
    sentences_list = []         # 每一个元素是一整个句子
    sentences_list_labels = []  # 每个元素是一整个句子的标签
    with open(path, 'r', encoding='UTF-8') as f:
        sentence_labels = []    # 每个元素是这个句子的每个单词的标签
        sentence = []           # 每个元素是这个句子的每个单词

        for line in f:
            line = line.strip()
            if not line:        # 如果遇到了空白行
                if sentence:    # 防止空白行连续多个，导致出现空白的句子
                    sentences_list.append(' '.join(sentence))
                    sentences_list_labels.append(' '.join(sentence_labels))

                    sentence = []
                    sentence_labels = []
                                # 创建新的句子的list，准备读入下一个句子
            else:
                res = line.split()
                assert len(res) == 4
                if res[0] == '-DOCSTART-':
                    continue
                sentence.append(res[0])
                sentence_labels.append(res[3])

        if sentence:            # 防止最后一行没有空白行，导致最后一句话录入不到
            sentences_list.append(sentence)
            sentences_list_labels.append(sentence_labels)
    return sentences_list[:length], sentences_list_labels[:length]

def build_vocab(sentences_list):
    ret = []
    for sentences in sentences_list:
        ret += [word for word in sentences.split()]
    return list(set(ret))

class mydataset(Dataset):
    def __init__(self, x : torch.Tensor, y : torch.Tensor, length_list):
        self.x = x
        self.y = y
        self.length_list = length_list
    def __getitem__(self, index):
        data = self.x[index]
        labels = self.y[index]
        length = self.length_list[index]
        return torch.tensor(data), torch.tensor(labels), torch.tensor(length)
        #return data, labels, length
    def __len__(self):
        return len(self.x)

def get_idx(word, d):
    if d[word] is not None:
        return d[word]
    else:
        return d['<unknown>']

def sentence2vector(sentence, d):
    return [get_idx(word, d) for word in sentence.split()]

def padding(x, max_length, d):
    length = 0
    for i in range(max_length - len(x)):
        x.append(d['<pad>'])
    return x

def get_dataloader(x, y, batch_size):
    inputs = [sentence2vector(s, word2idx) for s in x] # 每一个句子都转化成vector

    targets = [sentence2vector(s, tag2idx) for s in y]

    length_list = [len(sentence) for sentence in inputs]

    max_length = 0

    max_length = max(max(length_list), max_length)

    inputs = torch.tensor([padding(sentence, max_length, word2idx) for sentence in inputs])

    targets = torch.tensor([padding(sentence, max_length, tag2idx) for sentence in targets], dtype=torch.long)

    dataset = mydataset(inputs, targets, length_list)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloader, max_length

def pre_processing():
    x_train, y_train = read_data("data/conll2003/train.txt", 10000)
    x_test, y_test = read_data("data/conll2003/test.txt", 1000)
    d_x = build_vocab(x_train+x_test)
    d_y = build_vocab(y_train+y_test)
    word2idx = {d_x[i]: i for i in range(len(d_x))}
    tag2idx = {d_y[i]: i for i in range(len(d_y))}
    tag2idx["<START>"] = 3
    tag2idx["<STOP>"] = 4
    pad_idx = len(word2idx)
    word2idx['<pad>'] = pad_idx
    tag2idx['<pad>'] = len(tag2idx)
    vocab_size = len(word2idx)

    return word2idx, tag2idx, vocab_size

word2idx, tag2idx, vocab_size = pre_processing()
