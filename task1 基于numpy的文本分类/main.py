import numpy as np
import pandas as pd
from BagofWords import BagofWords
from Ngram import Ngram
from sklearn.model_selection import train_test_split
from model import softmax_regression
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_df = pd.read_csv('./data/train.tsv', sep='\t')
    X_data, y_data = train_df["Phrase"].values, train_df["Sentiment"].values
    # test_df = pd.read_csv(test_file, sep="\t")

    just_test = 1

    if just_test == 1:
        X_data = X_data[:20000]
        y_data = y_data[:20000]

    y = np.array(y_data).reshape((-1, 1))

    bag_of_words = BagofWords()
    Ngram = Ngram(ngram=(1, 2))
    X_Bow = bag_of_words.fit_transform(X_data)
    X_Gram = Ngram.fit_transform(X_data)

    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(X_Bow, y, test_size=0.2, random_state=42, stratify=y)   #按y中各类比例，分配给train和test
    X_train_Gram, X_test_Gram, y_train_Gram, y_test_Gram = train_test_split(X_Gram, y, test_size=0.2, random_state=42, stratify=y)

    epochs = 100
    bow_learning_rate = 1
    gram_learning_rate = 1

    model1 = softmax_regression()
    history = model1.fit(X_train_Bow, y_train_Bow, epochs=epochs, learning_rate=bow_learning_rate)
    plt.title('Bag of words')
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Bow train {} test {}".format(model1.score(X_train_Bow, y_train_Bow), model1.score(X_test_Bow, y_test_Bow)))

    model2 = softmax_regression()
    history = model2.fit(X_train_Gram, y_train_Gram, epochs=epochs, learning_rate=gram_learning_rate)
    plt.title('N-Gram')
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Gram train {} test {}".format(model2.score(X_train_Gram, y_train_Gram),
                                         model2.score(X_test_Gram, y_test_Gram)))




