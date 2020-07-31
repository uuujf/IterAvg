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

logdir = './logs/gd'
num_iters = 2000
lr = 0.005

def L1Path(model, lr, l1regu_list, num_iters):
    l1_path = np.zeros((len(l1regu_list), 2))

    for i in range(len(l1regu_list)):
        l1regu = l1regu_list[i]
        model.initialization()
        for _ in range(num_iters):
            loss = model.loss() + model.L1(l1regu)
            dw = -lr * grad(loss, model.w)[0]
            model.w.data += dw
        l1_path[i] = model.w.detach().numpy().copy()
        print ('L1: {:.3f}, Solution: [{:.4f},{:.4f}]'.format(l1regu, l1_path[i,0], l1_path[i,1]))

    return l1_path

def L2Path(model, l2regu_list):
    l2_path = np.zeros((len(l2regu_list), 2))

    for i in range(len(l2regu_list)):
        l2regu = l2regu_list[i]
        Sigma_hat = model.Sigma_np + l2regu * np.eye(2)
        Sigma_hat_inverse = np.linalg.inv(Sigma_hat)
        solution = np.matmul(Sigma_hat_inverse, np.matmul(model.Sigma_np, model.w_star_np.T))
        l2_path[i] = np.copy(solution).reshape(-1)

    return l2_path

if __name__ == '__main__':
     # logs
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.chdir(logdir)
    print(os.getcwd())

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = Basin().to(device)

    # l1 path
    l1regu_list = list(np.arange(0,1,0.01))
    print(len(l1regu_list))

    wl1 = L1Path(model, lr, l1regu_list, num_iters)
    np.save('l1_path', wl1)
    # wl1 = np.load('l1_path.npy')

    # l2 path
    l2regu_list = list(np.arange(0,5,0.01))
    print(len(l2regu_list))

    wl2 = L2Path(model, l2regu_list)
    np.save('l2_path', wl2)
    # wl2 = np.load('l2_path.npy')


    # GD path
    w = np.load('w_eta{}_regu{}.npy'.format(0.1, 0.1))

    # plot
    plt.figure(1)
    model.plot_contour(plt)

    plt.scatter(w[:,0], w[:,1], c='g', s=1.5, label='GD Path')
    plt.scatter(wl2[:,0], wl2[:,1], c='y', s=1.5, label='$\ell_2$-regularization Path')
    plt.scatter(wl1[:,0], wl1[:,1], c='m', s=1.5, label='$\ell_1$-regularization Path')
    
    plt.legend(loc='upper left', markerscale=8, fontsize=16)
    plt.savefig('reguPath.pdf')
