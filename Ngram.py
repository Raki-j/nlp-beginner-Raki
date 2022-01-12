import numpy as np

class Ngram:
    def __init__(self, ngram, do_lower_case=False):
        self.ngram = ngram   #[2,3,4]gram
        self.feature_map = {}  #特征矩阵
        self.do_lower_case = do_lower_case

    def fit_transform(self, sent_list):
        for gram in self.ngram:
            for sent in sent_list:
                if self.do_lower_case:
                    sent = sent.lower()
                sent = sent.split(" ")
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])  #用来生成一个个n-gram的词组
                    if feature not in self.feature_map:
                        self.feature_map[feature] = len(self.feature_map)
        n = len(sent_list)  #总句子数量
        m = len(self.feature_map)  #n元组的数量
        ngram_feature = np.zeros((n, m))
        for idx, sent in enumerate(sent_list): #取出下标和句子
            if self.do_lower_case:
                sent = sent.lower()
            sent = sent.split(" ")
            for gram in self.ngram:
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])
                    if feature in self.feature_map:
                        ngram_feature[idx][self.feature_map[feature]] = 1
        return ngram_feature
