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

def plotError(plt, w, what, wtilde):
    error1 = np.abs(w - what).sum(-1)
    error2 = np.abs(wtilde - what).sum(-1)

    plt.plot(range(len(error1)), error1, color='g', label='$|w_k-\\hat{w}_k|$')
    plt.plot(range(len(error2)), error2, color='k', label='$|\\tilde{w}_k - \\hat{w}_k|$')
    plt.yscale('log')
    # plt.ylim(bottom=1e-3)
    plt.xlabel('iteration', fontsize=16)
    plt.ylabel('abs error', fontsize=16)
    plt.legend(loc='center right', fontsize=16)

def test(model, dataloader):
    model.eval()
    acc = 0.0
    for x,y in dataloader:
        out = model(x)
        acc += accuracy(out, y).item()
    acc /= len(dataloader)
    return acc

def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res

class LogSaver(object):
    def __init__(self, logdir, logfile=None):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if logfile:
            self.saver = os.path.join(logdir, logfile)
        else:
            self.saver = os.path.join(logdir, str(datetime.now())+'.log')
        print('save logs at:', self.saver)

    def save(self, item, name=None):
        with open(self.saver, 'a') as f:
            if name:
                f.write('======'+name+'======\n')
                print('======'+name+'======')
            f.write(item+'\n')
            print(item)
