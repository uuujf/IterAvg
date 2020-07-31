import os
import numpy as np
from sklearn.utils import shuffle as skshuffle
import torch

class MNIST(object):
    def __init__(self, datadir, device):
        super(MNIST, self).__init__()
        self.n_classes = 10
        self.datadir = datadir
        self.device = device

        train = np.load(os.path.join(datadir, 'train.npz'))
        test = np.load(os.path.join(datadir, 'test.npz'))

        self.X_train = train['image'].astype(np.float32) / 255.0
        self.Y_train = train['label']
        self.X_test = test['image'].astype(np.float32) / 255.0
        self.Y_test = test['label']

        self.n_test = len(self.Y_test)
        self.n_train = len(self.Y_train)

    def __str__(self):
        return self.datadir

    def to_tensor(self, X, Y, device):
        X = torch.FloatTensor(X).to(device)
        Y = torch.LongTensor(Y).to(device)
        return X, Y

    def getTrainBatch(self, batch_size):
        mask = np.random.choice(self.n_train, batch_size, False)
        X, Y = self.X_train[mask], self.Y_train[mask]
        return self.to_tensor(X, Y, self.device)

    def getTestList(self, batch_size=500):
        n_batch = self.n_test // batch_size
        batch_list = []
        for i in range(n_batch):
            X, Y = self.X_test[batch_size*i:batch_size*(i+1)], self.Y_test[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, self.device)
            batch_list.append((X, Y))
        return batch_list

    def getTrainList(self, batch_size=1000, shuffle=True):
        n_batch = self.n_train // batch_size
        batch_list = []
        X_shuffle, Y_shuffle = self.X_train.copy(), self.Y_train.copy()
        if shuffle:
            X_shuffle, Y_shuffle = skshuffle(self.X_train, self.Y_train)
        for i in range(n_batch):
            X, Y = X_shuffle[batch_size*i:batch_size*(i+1)], Y_shuffle[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, self.device)
            batch_list.append((X, Y))
        return batch_list

if __name__ == '__main__':

    HOME = os.environ['HOME']
    datapath = os.path.join(HOME, 'datasets/MNIST/numpy')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MNIST(datapath, device)
    from IPython import embed; embed()
