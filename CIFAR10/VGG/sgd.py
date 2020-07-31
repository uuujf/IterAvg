import os
from datetime import datetime
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter

from cifar10 import CIFAR10
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 250])
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('--aug', action='store_true', default=True)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--logdir', type=str, default='logs/VGG')

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
testloader = dataset.getTestList(500)

# model
start_epoch = 0
lr = args.lr
from vgg import vgg16_bn
model = vgg16_bn().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weightdecay)
if args.resume:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    lr = checkpoint['lr']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.save("=> loaded checkpoint '{}'".format(args.resume))
logger.save(str(model), 'classifier')
logger.save(str(optimizer), 'optimizer')

# writer
writer = SummaryWriter(args.logdir)


# optimization
torch.backends.cudnn.benchmark = True
for i in range(start_epoch, args.epochs):
    # decay lr
    if i in args.schedule:
        lr *= 0.1
        logger.save('update lr: %f'%(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # train
    trainloader = dataset.getTrainList(args.batchsize, True, args.aug)
    train_loss, train_acc = train(model, criterion, optimizer, trainloader)
    writer.add_scalar('acc/train', train_acc, i)
    writer.add_scalar('loss/train', train_loss, i)
    writer.add_scalar('lr', lr, i)

    # evaluate
    test_loss, test_acc = test(model, criterion, testloader)
    writer.add_scalar('loss/test', test_loss, i)
    writer.add_scalar('acc/test', test_acc, i)
    writer.add_scalar('acc/diff', train_acc - test_acc, i)

    logger.save('Epoch:%d, Test [acc: %.2f, loss: %.4f], Train [acc: %.2f, loss: %.4f]' \
                % (i, test_acc, test_loss, train_acc, train_loss))

    # save
    state = {'epoch':i, 'lr':lr, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(state, args.logdir+'/epoch-'+str(i)+'.pth.tar')

writer.close()

end_time = datetime.now()
logger.save(str(end_time), 'end time')
logger.save(str(end_time - start_time), 'duration')
