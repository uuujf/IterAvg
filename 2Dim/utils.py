import torch
import numpy as np
import os

def genWeight(length, prob):
    weight = np.ones(length) * prob
    exponent = np.array(list(range(length)))
    weight = np.power(weight, exponent)
    weight = weight * (1-prob)
    return weight / weight.sum()

def averagePath(w, p):
    p = p.reshape(-1,1)
    wtilde = np.zeros_like(w)
    for i in range(len(w)):
        wi = w[:i+1]
        pi = p[:i+1]
        wtilde[i] = (wi * pi).sum(axis=0)
        if i % 50 == 0:
            print('In averaging: [{}/{}]'.format(i, len(w)))
    return wtilde

def plotPath(plt, model, w, what, wtilde):
    # plot contour
    model.plot_contour(plt)

    # plot paths
    plt.scatter(w[:,0], w[:,1], c='g', s=1.5, label='$w_k$')
    plt.scatter(what[:,0], what[:,1], c='b', s=1.5, label='$\hat{w}_k$')
    plt.scatter(wtilde[:,0], wtilde[:,1], c='r', s=1.5, label='$\\tilde{w}_k$')

    plt.legend(loc='upper left', markerscale=8, fontsize=16)