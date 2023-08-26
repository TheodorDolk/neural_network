import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt
from math import sqrt, e
from random import randint
from tqdm import tqdm


def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')

def scale(train_arr, test_arr):
    max_val = max([train_arr.max(), test_arr.max()])
    scaled_train = train_arr / max_val
    scaled_test = test_arr / max_val
    return scaled_train, scaled_test

def mse_cost_func(y_true, y_pred):
    data_length = y_true.shape[0]
    return

def sigmoid(value, derivative=False):
    if derivative:
        return sigmoid(value) * (1 - sigmoid(value))
    return 1 / (1 + e ** (-value))


def xavier_init_uniform(n_in, n_out):
    limit = np.sqrt(6.0 / (n_in + n_out))
    weights = np.random.uniform(-limit, limit, size=(n_in, n_out))
    return weights

class NeuralNetwork:
    def __init__(self, input, n_input, n_output, hidden_layers):
        self.input = input
        self.output = None
        self.h_layers = hidden_layers
        self.n_h_layers = len(self.h_layers)
        self.weights = []
        self.biases = []
        self.learning_rate = None
        self.n_epochs = None
        self.batch_size = None
        self.loss = 0
        self.n_input = n_input
        self.n_output = n_output

    def init_weights(self):
        # initiate weights for each layer
        for i, layer_size in enumerate(self.h_layers):
            if i == 0:
                b0 = xavier_init_uniform(self.n_input, 1)
                w_i_h0 = xavier_init_uniform(self.n_input, layer_size)
                self.weights.append(w_i_h0)
            elif i + 1 == self.n_h_layers:
                w_hn_o = xavier_init_uniform(layer_size, self.n_output)
                self.weights.append(w_hn_o)
            else:
                w_hn_hn = xavier_init_uniform(layer_size, self.h_layers[i + 1])
                self.weights.append(w_hn_hn)



    def forward_propagation(self):
        for i in range(self.n_h_layers + 2):
            


        pass

    def back_propagation(self):
        pass

    def train(self):
        pass

def main():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # scale data
    X_train_s, X_test_s = scale(X_train, X_test)
    y_train_s, y_test_s = scale(y_train, y_test)

    # create neural network
    neural = NeuralNetwork(2, 784, 10,X_train_s, X_test_s, y_train_s, y_test_s)



    pass

if __name__ == "__main__":
    main()
