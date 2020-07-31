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

logdir = './LinReg-logs/sgd'
num_iters = 501
# bs = 60000
bs = 500
l2regu = 4.0
eta = 0.01
gamma = 1 / (1/eta + l2regu)

w_name = 'linear-bs{}-eta{}-regu{}.npy'.format(bs, eta, 0.0)
what_name = 'linear-bs{}-gamma{}-regu{}.npy'.format(bs, gamma, l2regu)
fig_name = 'linear-bs{}-eta{}-regu{}.pdf'.format(bs, eta, l2regu)

def SGDPath(model, dataset, lr, l2regu, num_iters, batchsize):
    testloader = dataset.getTestList(10000)
    opt_path = np.zeros((num_iters, model.n_weights))
    model.initialization(0.0)

    for i in range(num_iters):
        opt_path[i] = model.weight.detach().numpy().copy()

        x, y = dataset.getTrainBatch(batchsize)
        out = model(x)
        loss = model.MSE(out, y) + model.L2(l2regu)
        dw = -lr * grad(loss, model.weight)[0]
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

    # model
    model = Linear().to(device)

    # generate path
    w = SGDPath(model, dataset, eta, 0, num_iters, bs)
    np.save(w_name, w)

    what = SGDPath(model, dataset, gamma, l2regu, num_iters, bs)
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
