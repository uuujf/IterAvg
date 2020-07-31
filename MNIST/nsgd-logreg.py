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

logdir = './LogReg-logs/nsgd'
num_iters = 501
alpha0 = 1.0
alpha = alpha0
bs = 60000
# bs = 500
l2regu = 4.0
eta = 0.01
gamma = 1 / (1/eta + l2regu)
tau = (1 - np.sqrt(eta*alpha)) / (1 + np.sqrt(eta*alpha))
tau_hat = (1 - np.sqrt(gamma*(alpha+l2regu))) / (1 + np.sqrt(gamma*(alpha+l2regu)))

w_name = 'NSGD-linear-bs{}-alpha{}-eta{}-regu{}.npy'.format(bs, alpha, eta, 0.0)
what_name = 'NSGD-linear-bs{}-alpha{}-gamma{}-regu{}.npy'.format(bs, alpha, gamma, l2regu)
fig_name = 'NSGD-linear-bs{}-alpha{}-eta{}-regu{}.pdf'.format(bs, alpha, eta, l2regu)

def NSGDPath(model, dataset, lr, tau, l2regu, num_iters, batchsize):
    testloader = dataset.getTestList(10000)
    opt_path = np.zeros((num_iters, model.n_weights))
    model.initialization(0.0)

    w0, w = torch.zeros_like(model.weight.data), torch.zeros_like(model.weight.data)

    opt_path[0] = model.weight.detach().numpy().copy()
    for i in range(num_iters-1):
        opt_path[i+1] = model.weight.detach().numpy().copy()

        v = w + tau*(w-w0)
        model.weight.data = v + 0.0
        x, y = dataset.getTrainBatch(batchsize)
        out = model(x)
        loss = model.CE(out, y) + model.L2(l2regu)
        dw = - lr * grad(loss, model.weight)[0]
        train_loss = loss.item()
        w0 = w + 0.0
        w = v + dw
        model.weight.data = w + 0.0

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
    w = NSGDPath(model, dataset, eta, tau, alpha0, num_iters, bs)
    np.save(w_name, w)

    what = NSGDPath(model, dataset, gamma, tau_hat, alpha0+l2regu, num_iters, bs)
    np.save(what_name, what)

    # w = np.load(w_name)
    # what = np.load(what_name)

    # generate weight
    prob = (1 - np.sqrt(gamma * (alpha + l2regu))) / (1 - np.sqrt(eta * alpha))
    p = np.ones(len(w)) * prob
    exponent = np.array(list(range(-2, len(w)-2)))
    p = np.power(p, exponent)
    p *= (1-prob) * gamma / eta
    p[0] = 1 - gamma / eta / prob
    p /= p.sum()

    wtilde = averagePath(w, p)

    # plot
    plt.figure(1)
    plotError(plt, w, what, wtilde)
    plt.savefig(fig_name)
