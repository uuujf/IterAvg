import os
import numpy as np
import torch
from torch.autograd import grad

from mnist import MNIST
from model import Linear, LinearBias


HOME = os.environ['HOME']
dataset = MNIST(os.path.join(HOME, 'datasets/MNIST/numpy'), 60000)
train_loader = dataset.getTrainList(60000, False)

model = LinearBias()


X, y = train_loader[0]
alpha0 = 0.1


# def onehot(label, num_classes=10):
#     onehot = torch.eye(num_classes)[label]
#     return onehot

# out = model(X)
# loss = model.MSE(out, onehot(y), alpha0)

# dw = grad(loss, model.weight, create_graph=True)[0]
# Hess = []
# for i in range(len(dw)):
#     ggi = grad(dw[i], model.weight, retain_graph=True)[0]
#     Hess.append(ggi.clone())
#     print(i)
# Hess = torch.stack(Hess, dim=0)

# Hess = Hess.numpy()

X = X.numpy()

Hess = np.matmul(X.T, X) / len(X) + alpha0 * np.eye(784)

Hess_full = np.kron(np.eye(10), Hess)

np.save('Hess', Hess)
np.save('HessFull', Hess_full)

Hess_inv = np.linalg.inv(Hess)

Hess_full_inv = np.kron(np.eye(10), Hess_inv)
np.save('Hess_inv', Hess_inv)
np.save('HessFull_inv', Hess_full_inv)

