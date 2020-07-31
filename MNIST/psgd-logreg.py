import os
import numpy as np
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from utils import *
from mnist import MNIST
from model import Linear, LinearBias

logdir = './LogReg-logs/psgd'
num_iters = 501
bs = 60000
# bs = 500
alpha0 = 1.0
l2regu = 4.0
eta = 0.01
gamma = 1 / (1/eta + l2regu)

w_name = 'PSGD-linear-bs{}-a0{}-eta{}-regu{}.npy'.format(bs, alpha0, eta, 0.0)
what_name = 'PSGD-linear-bs{}-a0{}-gamma{}-regu{}.npy'.format(bs, alpha0, gamma, l2regu)
fig_name = 'PSGD-linear-bs{}-a0{}-eta{}-regu{}.pdf'.format(bs, alpha0, eta, l2regu)

def PSGDPath(model, dataset, lr, l2regu, num_iters, batchsize, Q, Q_inv):
    testloader = dataset.getTestList(10000)
    opt_path = np.zeros((num_iters, model.n_weights))
    model.initialization(0.0)

    for i in range(num_iters):
        opt_path[i] = model.weight.detach().numpy().copy()

        x, y = dataset.getTrainBatch(batchsize)
        out = model(x)
        loss = model.CE(out, y) + model.generalL2(Q, l2regu)
        dw = -lr * grad(loss, model.weight)[0]
        dw = torch.matmul(dw, Q_inv)
        train_loss = loss.item()
        model.weight.data += dw

        if i % 50 == 0:
            acc = test(model, testloader)
            print ('Step [{}/{}], Loss: {:.8f}, Acc: {:.2f} %'.format(i, num_iters, train_loss, acc))

    return opt_path


if __name__ == "__main__":
    # logs
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.chdir(logdir)
    print(os.getcwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    HOME = os.environ['HOME']
    dataset = MNIST(os.path.join(HOME, 'datasets/MNIST/numpy'), device)

    # Q
    Q = np.load('HessFull.npy')
    Q_inv = np.load('HessFull_inv.npy')
    Q = torch.FloatTensor(Q).to(device)
    Q_inv = torch.FloatTensor(Q_inv).to(device)

    # model
    model = Linear().to(device)

    # generate path
    w = PSGDPath(model, dataset, eta, alpha0, num_iters, bs, Q, Q_inv)
    np.save(w_name, w)

    what = PSGDPath(model, dataset, gamma, alpha0+l2regu, num_iters, bs, Q, Q_inv)
    np.save(what_name, what)

    # w = np.load(w_name)
    # what = np.load(what_name)

    # generate weight
    p = genWeight(len(w), gamma/eta)
    wtilde = averagePath(w, p)

    # plot
    plt.figure(1)
    plotError(plt, w, what, wtilde)
    plt.savefig(fig_name)
