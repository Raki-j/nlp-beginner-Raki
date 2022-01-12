import numpy as np
import pandas as pd

class softmax_regression():
    def __init__(self, epochs = 10, learning_rate = 0.01):
        self.batch_size = None
        self.num_features = None
        self.w = None
        self.learning_rate = learning_rate
        self.num_classes = None
        self.epochs = epochs

    def fit(self, X, y, learning_rate=0.01, epochs=10, num_classes=5):
        '''

        :param X: [batch_size, num_features]
        :param y: [batch_size, 1]
        :param w: [num_classes, num_features]
        :return:

        '''
        self.__init__(epochs, learning_rate=learning_rate)
        self.batch_size, self.num_features = X.shape
        self.num_classes = num_classes
        self.w = np.random.randn(self.num_classes, self.num_features)

        y_one_hot = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            y_one_hot[i][y[i]] = 1 #把y所属的类标记为1

        loss_history = []

        for i in range(epochs):
            loss = 0
            probs = X.dot(self.w.T)
            probs = softmax(probs)
            for i in range(self.batch_size):
                loss -= np.log(probs[i][y[i]])
            weight_update = np.zeros_like(self.w)
            for i in range(self.batch_size):
                weight_update += X[i].reshape(1, self.num_features).T.dot((y_one_hot[i] - probs[i]).reshape(1, self.num_classes)).T
                #拿出X的第i行
            self.w += weight_update * self.learning_rate / self.batch_size

            loss /= self.batch_size
            loss_history.append(loss)
            if i % 10 == 0:
                print("epoch {} loss {}".format(i, loss))
        return loss_history

    def predict(self, X):
        prob = softmax(X.dot(self.w.T))
        return prob.argmax(axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred.reshape(y.shape) == y) / y.shape[0]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    # 稳定版本的softmax，对z的每一行进行softmax
    z -= np.max(z, axis=1, keepdims=True)  # 先减去该行的最大值
    z = np.exp(z)
    z /= np.sum(z, axis=1, keepdims=True)
    return z


