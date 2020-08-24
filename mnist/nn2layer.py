import sys, os
sys.path.append(os.pardir)
import numpy as np

import utils.activation as af
import utils.gradient as gf
import utils.loss as lf

class TwoLayernet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = af.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = af.softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return lf.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = gf.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = gf.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = gf.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = gf.numerical_gradient(loss_W, self.params['b2'])

        return grads

