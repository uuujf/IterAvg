import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Basin(nn.Module):
    def __init__(self):
        super(Basin, self).__init__()
        theta = math.pi/3
        sigma = np.array([[0.1, 0.0], [0.0, 1.0]])
        K1 = math.cos(theta)
        K2 = math.sin(theta)
        U = np.array([[K1, -K2], [K2, K1]]) # 2x2
        self.Sigma_np = np.matmul(np.matmul(U, sigma), U.T) # 2x2
        self.w_star_np = np.array([[1,1]]) # 1x2

        self.Sigma = torch.FloatTensor(self.Sigma_np)
        self.w_star = torch.FloatTensor(self.w_star_np)
        self.w = torch.nn.Parameter(torch.zeros(1,2), requires_grad=True)

        self.Q_np = np.copy(self.Sigma_np)
        self.Q_inv_np = np.linalg.inv(self.Q_np)
        self.Q = torch.FloatTensor(self.Q_np)
        self.Q_inv = torch.FloatTensor(self.Q_inv_np)
    
    def initialization(self):
        self.w.data *= 0.0

    def loss(self):
        diff = self.w - self.w_star
        return (torch.matmul(diff, self.Sigma) * diff).sum() * 0.5
    
    def L2(self, l2regu=1.0):
        return (self.w**2).sum() * 0.5 * l2regu
    
    def generalL2(self, l2regu=1.0):
        return (torch.matmul(self.w, self.Q) * self.w).sum() * 0.5 * l2regu

    def L1(self, l1regu=1.0):
        return self.w.abs().sum() * l1regu

    def loss_np(self, w):
        # w: N*2
        diff = w - self.w_star_np
        return (np.matmul(diff, self.Sigma_np) * diff).sum(axis=1) * 0.5

    def plot_contour(self, plt):
        w1 = np.arange(-0.3, 1.3, 0.02)
        w2 = np.arange(-0.3, 1.3, 0.02)
        W1, W2 = np.meshgrid(w1, w2)
        W = np.array([[w1, w2] for w1, w2 in zip(np.ravel(W1), np.ravel(W2))])
        s = self.loss_np(W)
        S = s.reshape(W1.shape)

        c = plt.contour(W1, W2, S, 8)
        plt.clabel(c, inline=True, fontsize=10)
        # plt.xlabel('$w_1$', fontsize=16)
        # plt.ylabel('$w_2$', fontsize=16)