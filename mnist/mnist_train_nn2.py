import numpy as np
import matplotlib.pyplot as plt

from mnist import load_mnist
from nn2layer import TwoLayernet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# super parameters
step_size = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

from tensorflow.python.keras import backend as K
print("checking if GPU available", K._get_available_gpus())

network = TwoLayernet(input_size=784, hidden_size=50, output_size=10)

for i in range(step_size):
    #Get mini-batch 
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #Calcuate gradient
    grad = network.numerical_gradient(x_batch, t_batch)

    #Update parameter
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)


x = np.arange(0, step_size, 1)
y = train_loss_list

plt.plot(x, y, label="loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MNIST")
plt.legend()

plt.savefig()("mnist_loss.jpeg", "JPEG")
