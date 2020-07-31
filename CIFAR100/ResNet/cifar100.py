import os
import numpy as np
from sklearn.utils import shuffle as skshuffle
import torch

class CIFAR100(object):
    def __init__(self, datadir, device):
        super(CIFAR100, self).__init__()
        self.n_classes = 100
        self.datadir = datadir
        self.device = device

        train = np.load(os.path.join(datadir, 'train.npz'))
        test = np.load(os.path.join(datadir, 'test.npz'))
        self.X_train = train['image']
        self.Y_train = train['label']
        self.X_test = test['image']
        self.Y_test = test['label']

        self.n_test = len(self.Y_test)
        self.n_train = len(self.Y_train)

        self.trans()

    def __str__(self):
        return self.datadir
        # return 'CIFAR10\nnum_train: %d\nnum_test: %d' % (self.n_train, self.n_test)

    def transpose(self):
        self.X_train = np.transpose(self.X_train, [0,3,1,2])
        self.X_test = np.transpose(self.X_test, [0,3,1,2])

    def normalize(self):
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def trans(self):
        self.transpose()
        self.normalize()

    def getTrainBatch(self, batch_size, aug=False):
        mask = np.random.choice(self.n_train, batch_size, False)
        X, Y = self.X_train[mask], self.Y_train[mask]
        if aug:
            X = self.random_crop(X)
            X = self.horizontal_flip(X)
        return self.to_tensor(X, Y, self.device)

    def getTestList(self, batch_size=500):
        n_batch = self.n_test // batch_size
        batch_list = []
        for i in range(n_batch):
            X, Y = self.X_test[batch_size*i:batch_size*(i+1)], self.Y_test[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, self.device)
            batch_list.append((X, Y))
        return batch_list

    def getTrainList(self, batch_size=1000, shuffle=True, aug=False):
        n_batch = self.n_train // batch_size
        batch_list = []
        X_shuffle, Y_shuffle = self.X_train.copy(), self.Y_train.copy()
        if shuffle:
            X_shuffle, Y_shuffle = skshuffle(self.X_train, self.Y_train)
        for i in range(n_batch):
            X, Y = X_shuffle[batch_size*i:batch_size*(i+1)], Y_shuffle[batch_size*i:batch_size*(i+1)]
            if aug:
                X = self.random_crop(X)
                X = self.horizontal_flip(X)
            X, Y = self.to_tensor(X, Y, self.device)
            batch_list.append((X, Y))
        return batch_list

    def random_crop(self, X, pad=2):
        b, c, h, w = X.shape
        X = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
        X_randcrop = []
        for x in X:
            top = np.random.randint(0, 2*pad)
            left = np.random.randint(0, 2*pad)
            X_randcrop.append(x[:,top:top+h,left:left+w])
        X_randcrop = np.stack(X_randcrop, axis=0)
        return X_randcrop

    def horizontal_flip(self, X, rate=0.5):
        X_flip = []
        for x in X:
            if np.random.rand() < rate:
                x = x[:,:,::-1]
            X_flip.append(x)
        X_flip = np.stack(X_flip, axis=0)
        return X_flip

    def to_tensor(self, X, Y, device):
        X = torch.FloatTensor(X).to(device)
        Y = torch.LongTensor(Y).to(device)
        return X, Y

if __name__ == '__main__':
    HOME = os.environ['HOME']
    datapath = os.path.join(HOME, 'datasets/CIFAR100/numpy')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CIFAR100(datapath, device)
    from IPython import embed; embed()
