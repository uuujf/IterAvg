import torch
import torch.nn as nn
import torch.nn.functional as F

def onehot(label, num_classes=10):
    onehot = torch.eye(num_classes)[label]
    return onehot

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.n_weights = 7840
        self.weight = torch.nn.Parameter(torch.zeros(self.n_weights), requires_grad=True)

    def initialization(self, init=0):
        self.weight.data.normal_(0.0, init)

    def forward(self, x):
        fc1_weight = self.weight[:7840].view(10,784)
        out = x
        out = F.linear(out, fc1_weight, bias=None)
        return out

    def MSE(self, output, target):
        target = onehot(target)
        mse_loss = (output-target).pow(2).sum(dim=1).mean(dim=0) / 2.0
        return mse_loss

    def CE(self, output, target):
        output = F.log_softmax(output, dim=-1)
        bs = target.shape[0]
        loss = - output[range(bs), target]
        loss = loss.mean()
        return loss

    def L2(self, l2regu=1.0):
        return self.weight.pow(2).sum() * 0.5 * l2regu
    
    def generalL2(self, Q, l2regu=1.0):
        return (torch.matmul(self.weight, Q) * self.weight).sum() * 0.5 * l2regu

class LinearBias(nn.Module):
    def __init__(self):
        super(LinearBias, self).__init__()
        self.n_weights = 7850
        self.weight = torch.nn.Parameter(torch.zeros(self.n_weights), requires_grad=True)

    def initialization(self, init=0):
        self.weight.data.normal_(0.0, init)

    def forward(self, x):
        fc1_weight = self.weight[:7840].view(10,784)
        fc1_bias = self.weight[7840:7850].view(10)
        out = x
        out = F.linear(out, fc1_weight, bias=fc1_bias)
        return out

    def MSE(self, output, target):
        target = onehot(target)
        mse_loss = (output-target).pow(2).sum(dim=1).mean(dim=0) / 2.0
        return mse_loss

    def CE(self, output, target):
        output = F.log_softmax(output, dim=-1)
        bs = target.shape[0]
        loss = - output[range(bs), target]
        loss = loss.mean()
        return loss
        
    def L2(self, l2regu=1.0):
        return self.weight.pow(2).sum() * 0.5 * l2regu

    def generalL2(self, Q, l2regu=1.0):
        return (torch.matmul(self.weight, Q) * self.weight).sum() * 0.5 * l2regu