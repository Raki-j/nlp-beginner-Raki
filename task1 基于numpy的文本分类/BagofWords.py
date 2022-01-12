import numpy as np

class BagofWords:
    def __init__(self, do_lower_case=False):
        self.vocab = {} #建立一个词表
        self.do_lower_case = do_lower_case

    def fit_transform(self, sent_list):
        for sent in sent_list:   #每一个句子
            if self.do_lower_case:  #是否要转化为小写字母
                sent = sent.lower()
            words = sent.strip().split(" ")  #把每句话拆成独立的单词
            for word in words: #对于每个单词，若不在词表中，则添加进词表
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        vocab_size = len(self.vocab) #词表长度
        bag_of_words_features = np.zeros((len(sent_list), vocab_size)) # 每一行代表一条语句，对应有词表中哪些词
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words: #如果该句子中有这个词，则在特征矩阵中+1
                bag_of_words_features[idx][self.vocab[word]] += 1
        return bag_of_words_features
