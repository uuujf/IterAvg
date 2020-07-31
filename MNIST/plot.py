import os
import numpy as np
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from utils import *

alpha = 1.0
l2regu = 4.0
eta = 0.01
gamma = 1 / (1/eta + l2regu)

logdir = './LogReg-logs/nsgd'
os.chdir(logdir)
print(os.getcwd())

# w_name = 'linear-bs{}-eta{}-regu{}.npy'.format(60000, eta, 0.0)
# what_name = 'linear-bs{}-gamma{}-regu{}.npy'.format(60000, gamma, l2regu)
# v_name = 'linear-bs{}-eta{}-regu{}.npy'.format(500, eta, 0.0)
# vhat_name = 'linear-bs{}-gamma{}-regu{}.npy'.format(500, gamma, l2regu)

w_name = 'linear-bs{}-alpha{}-eta{}-regu{}.npy'.format(60000, alpha, eta, 0.0)
what_name = 'linear-bs{}-alpha{}-gamma{}-regu{}.npy'.format(60000, alpha, gamma, l2regu)
v_name = 'linear-bs{}-alpha{}-eta{}-regu{}.npy'.format(500, alpha, eta, 0.0)
vhat_name = 'linear-bs{}-alpha{}-gamma{}-regu{}.npy'.format(500, alpha, gamma, l2regu)

w = np.load(w_name)
what = np.load(what_name)
v = np.load(v_name)
vhat = np.load(vhat_name)

# generate weight
prob = (1 - np.sqrt(gamma * (alpha + l2regu))) / (1 - np.sqrt(eta * alpha))
p = np.ones(len(w)) * prob
exponent = np.array(list(range(-2, len(w)-2)))
p = np.power(p, exponent)
p *= (1-prob) * gamma / eta
p[0] = 1 - gamma / eta / prob
p /= p.sum()
# p = genWeight(len(w), gamma/eta)

# average
wtilde = averagePath(w, p)
vtilde = averagePath(v, p)

# plot
plt.figure(1)
error1 = np.abs(w - what).sum(-1)
error2 = np.abs(wtilde - what).sum(-1)
error3 = np.abs(v - vhat).sum(-1)
error4 = np.abs(vtilde - vhat).sum(-1)

plt.plot(range(len(error1)), error1, ls='-', color='g', label='$|w_k-\\hat{w}_k|$ (NGD)')
plt.plot(range(len(error2)), error2, ls='-', color='k', label='$|\\tilde{w}_k - \\hat{w}_k|$ (NGD)')

plt.plot(range(len(error3)), error3, ls='--', color='g', label='$|w_k-\\hat{w}_k|$ (NSGD)')
plt.plot(range(len(error4)), error4, ls='--', color='k', label='$|\\tilde{w}_k - \\hat{w}_k|$ (NSGD)')

plt.yscale('log')
# plt.ylim(bottom=1e-3)
plt.xlabel('iteration', fontsize=16)
plt.ylabel('abs error', fontsize=16)
plt.legend(loc='upper right', fontsize=16)


plt.savefig('plot.pdf')