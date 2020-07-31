import os
from datetime import datetime
import argparse
import numpy as np
import torch

from cifar10 import CIFAR10
from utils import *
from swa import SWA

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--prob', type=float, default=0.999)
parser.add_argument('--resume', type=str, default='logs/VGG')
parser.add_argument('--logdir', type=str, default='logs/VGG-avg')

def applyAveragingOnline(ckpts_files, weight):
    ckpt = torch.load(ckpt_files[0])['model']
    ckpt_ave = ckpt.copy()
    for key in ckpt_ave.keys():
        ckpt_ave[key] = torch.zeros_like(ckpt_ave[key]) * 0.0

    for i in range(len(ckpts_files)):
        ckpt = torch.load(ckpt_files[i])['model']
        for key in ckpt_ave.keys():
            ckpt_ave[key] += ckpt[key] * weight[i]

    return ckpt_ave

def fixBN(model, dataloader):
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    opt = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
    opt.bn_update(dataloader, model)
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    start_time = datetime.now()

    logger = LogSaver(args.logdir)
    logger.save(str(args), 'args')
    logger.save(str(start_time), 'start time')

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.save(str(device), 'device')

    # data
    HOME = os.environ['HOME']
    datadir = os.path.join(HOME, 'data/datasets/CIFAR10/numpy')
    dataset = CIFAR10(datadir, device)
    logger.save(str(dataset), 'dataset')
    testloader = dataset.getTestList(100)
    trainloader = dataset.getTrainList(100, shuffle=False, aug=False)

    # model
    from vgg import vgg16_bn
    model = vgg16_bn().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ckpt
    ckpt_files = []
    for i in range(args.start_epoch, args.epochs, args.step):
        ckpt_file = os.path.join(args.resume, 'epoch-'+str(i)+'.pth.tar')
        ckpt_files.append(ckpt_file)

    # weight
    weight = genWeight(len(ckpt_files), args.prob)

    # average
    ckpt_ave = applyAveragingOnline(ckpt_files, weight)
    # end_time = datetime.now()
    # logger.save(str(end_time - start_time), 'IO time')

    model.load_state_dict(ckpt_ave)
    fixBN(model, trainloader)
    # end_time = datetime.now()
    # logger.save(str(end_time - start_time), 'BN time')

    # evaluate
    test_loss, test_acc = test(model, criterion, testloader)

    logger.save('[acc: %.2f, loss: %.4f]' % (test_acc, test_loss), 'Test')

    end_time = datetime.now()
    logger.save(str(end_time), 'end time')
    logger.save(str(end_time - start_time), 'duration')
