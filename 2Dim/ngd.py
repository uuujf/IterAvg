import os
import math
import numpy as np
import torch
from torch.autograd import grad
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from model import Basin

logdir = './logs/ngd'
num_iters = 500
alpha = 0.05

l2regu = 0.1
eta = 0.1
gamma = 1 / (1/eta + l2regu)
tau = (1 - np.sqrt(eta*alpha)) / (1 + np.sqrt(eta*alpha))
tau_hat = (1 - np.sqrt(gamma*(alpha+l2regu))) / (1 + np.sqrt(gamma*(alpha+l2regu)))

w_name = 'w_alpha{}_eta{}_regu{}.npy'.format(eta, alpha, l2regu)
what_name = 'what_alpha{}_eta{}_regu{}.npy'.format(eta, alpha, l2regu)
fig_name = 'eta{}_alpha{}_regu{}.pdf'.format(eta, alpha, l2regu)


def NGDPath(model, lr, tau, l2regu, num_iters):
    opt_path = np.zeros((num_iters, 2))
    model.initialization()

    w0, w = torch.zeros_like(model.w), torch.zeros_like(model.w)

    opt_path[0] = w0.reshape(2) + 0.0
    for i in range(num_iters-1):
        opt_path[i+1] = model.w.detach().numpy().copy()

        v = w + tau*(w-w0)
        model.w.data = v + 0.0
        loss = model.loss() + model.L2(l2regu)
        dw = -lr * grad(loss, model.w)[0]
        w0 = w + 0.0
        w = v + dw
        model.w.data = w + 0.0

        model.w.data += dw

        if i % 50 == 0:
            print ('Step [{}/{}], Location: [{:.4f},{:.4f}]'.format(i, num_iters, opt_path[i,0], opt_path[i,1]))

    return opt_path


if __name__ == "__main__":
    # logs
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.chdir(logdir)
    print(os.getcwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = Basin().to(device)

    # paths
    w = NGDPath(model, eta, tau, 0, num_iters)
    np.save(w_name, w)

    what = NGDPath(model, gamma, tau_hat, l2regu, num_iters)
    np.save(what_name, what)

    # # load
    # w = np.load(w_name)
    # what = np.load(what_name)

    # gen weight
    prob = (1 - np.sqrt(gamma * (alpha + l2regu))) / (1 - np.sqrt(eta * alpha))
    p = np.ones(len(w)) * prob
    exponent = np.array(list(range(-2, len(w)-2)))
    p = np.power(p, exponent)
    p *= (1-prob) * gamma / eta
    p[0] = 1 - gamma / eta / prob
    p /= p.sum()
    # p = genWeight(len(w), gamma/eta)

    wtilde = averagePath(w, p)

    # plot
    plt.figure(1)
    # plotPath(plt, model, w, what, wtilde)

    # plot contour
    model.plot_contour(plt)

    # plot paths
    plt.scatter(w[:,0], w[:,1], c='g', s=1.5, label='NGD Path $w_k$')
    plt.scatter(what[:,0], what[:,1], c='b', s=1.5, label='Regularized NGD Path $\hat{w}_k$')
    plt.scatter(wtilde[:,0], wtilde[:,1], c='r', s=1.5, label='Averaged NGD Path $\\tilde{w}_k$')

    plt.legend(loc='upper left', markerscale=8, fontsize=16)

    plt.savefig(fig_name)
