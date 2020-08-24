# coding: utf-8
import sys, os
import numpy as np
import pickle
from mnist import load_mnist
sys.path.append(os.pardir)
from utils.activition import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("./datasets/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    print(W1.shape)
    print(W2.shape)
    print(W3.shape)
    print(b1.shape)
    print(b2.shape)
    print(b3.shape)
    a1 = np.dot(x, W1) + b1
    print(a1)
    z1 = sigmoid(a1)
    print(z1)
    a2 = np.dot(z1, W2) + b2
    print(a2)
    z2 = sigmoid(a2)
    print(z2)
    a3 = np.dot(z2, W3) + b3
    print(a3)
    y = softmax(a3)
    print(y)

    return y


x, t = get_data()
print(x.shape)
print(x.shape[0])
network = init_network()
#predict(network, x[0])

training_size = x.shape[0]
batch_size = 10

batch_mask = np.random.choice(training_size, batch_size)
x_batch = x[batch_mask]
print(x_batch.shape)
t_batch = t[batch_mask]
print(t_batch.shape)
