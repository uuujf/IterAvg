import os
import math
import numpy as np
import torch
from torch.autograd import grad
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from model import Basin

logdir = './logs/gd'
num_iters = 500

l2regu = 0.1
eta = 0.1
gamma = 1 / (1/eta + l2regu)

w_name = 'w_eta{}_regu{}.npy'.format(eta, l2regu)
what_name = 'what_eta{}_regu{}.npy'.format(eta, l2regu)
fig_name = 'eta{}_regu{}.pdf'.format(eta, l2regu)

def GDPath(model, lr, l2regu, num_iters):
    opt_path = np.zeros((num_iters, 2))
    model.initialization()
    for i in range(num_iters):
        opt_path[i] = model.w.detach().numpy().copy()
        
        loss = model.loss() + model.L2(l2regu)
        dw = -lr * grad(loss, model.w)[0]
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
    w = GDPath(model, eta, 0, num_iters)
    np.save(w_name, w)

    what = GDPath(model, gamma, l2regu, num_iters)
    np.save(what_name, what)

    # # load
    # w = np.load(w_name)
    # what = np.load(what_name)

    # gen weight
    p = genWeight(len(w), gamma/eta)
    wtilde = averagePath(w, p)

    # plot
    plt.figure(1)
    # plotPath(plt, model, w, what, wtilde)

    # plot contour
    model.plot_contour(plt)

    # plot paths
    plt.scatter(w[:,0], w[:,1], c='g', s=1.5, label='GD Path $w_k$')
    plt.scatter(what[:,0], what[:,1], c='b', s=1.5, label='Regularized GD Path $\hat{w}_k$')
    plt.scatter(wtilde[:,0], wtilde[:,1], c='r', s=1.5, label='Averaged GD Path $\\tilde{w}_k$')

    plt.legend(loc='upper left', markerscale=8, fontsize=16)

    plt.savefig(fig_name)
